import 'dotenv/config';
import { Telegraf } from 'telegraf';
import { message } from 'telegraf/filters';
import * as fs from 'fs/promises';
import * as path from 'path';
import Fuse, { type IFuseOptions } from 'fuse.js';
import OpenAI from 'openai';
import { z } from 'zod';

// Constants
const LAST_UPDATE_ID_FILE = path.resolve(process.cwd(), 'last_update_id.txt');
const DOCS_URL = 'https://orderly.network/docs/llms-full.txt';
const DOCS_BASE_URL = 'https://orderly.network/docs';
const MAX_FUSE_RESULTS = 7;
const MAX_DOC_CONTEXT_CHARACTERS = 10000;
const MAX_KB_CONTEXT_CHARACTERS = 5000;
const MAX_KNOWLEDGE_RESULTS = 15;
const KNOWLEDGE_FILE_PATH = path.resolve(process.cwd(), 'knowledge.json');
const MAX_HISTORY_MESSAGES = 10;

// Zod Schema for Classification Response
const RequestTypeEnum = z.enum([
  'documentation_query',
  'bot_related_inquiry',
  'broker_id_setup_inquiry',
  'unrelated_query',
]);

const ClassificationResponseSchema = z.object({
  requestType: RequestTypeEnum,
});

// Knowledge Base Item Interface
interface KnowledgeItem {
  question: string;
  answer: string;
  last_referenced_date: string;
}

// Global state
let rawDocumentationContent: string = '';
let preparedDocChunks: string[] = [];
let fuseInstance: Fuse<string> | null = null;
let knowledgeBase: KnowledgeItem[] = [];
let knowledgeFuseInstance: Fuse<KnowledgeItem> | null = null;
let aiClient: OpenAI | null = null;
const chatHistories: Record<number, OpenAI.Chat.Completions.ChatCompletionMessageParam[]> = {};

const token = process.env['TELEGRAM_BOT_TOKEN'];
if (!token) {
  throw new Error('TELEGRAM_BOT_TOKEN must be provided!');
}

function createOpenAIClient(): OpenAI {
  const apiKey = process.env['CEREBRAS_API_KEY'];
  if (!apiKey) {
    throw new Error('CEREBRAS_API_KEY environment variable is not set');
  }
  return new OpenAI({
    apiKey,
    baseURL: process.env['CEREBRAS_API_URL'] || 'https://api.cerebras.ai/v1',
  });
}

const bot = new Telegraf(token);

function preprocessDocs(docString: string): string[] {
  const allRelevantParagraphs: string[] = [];
  const sectionRegex = /#\s*([^\n]+)\nSource:\s*([^\n]+)\n([\s\S]*?)(?=\n#\s|$)/g;
  let currentSearchString = docString;
  let match;
  while ((match = sectionRegex.exec(currentSearchString)) !== null) {
    const header = match[1] ? match[1].trim() : '';
    const sourceUrl = match[2] ? match[2].trim() : '';
    const content = match[3] || '';
    if (header && sourceUrl) {
      if (!sourceUrl.includes('tech-doc')) {
        if (content.trim().length > 0) {
          allRelevantParagraphs.push(`Section: ${header}\n${content}`);
        }
      }
    }
    currentSearchString = currentSearchString.substring(match.index + match[0].length);
    sectionRegex.lastIndex = 0;
  }
  if (allRelevantParagraphs.length === 0) {
    console.log('Warning: preprocessDocs resulted in 0 items. Fuse.js may not be effective.');
  }
  return allRelevantParagraphs;
}

async function initializeDocumentation(): Promise<void> {
  try {
    console.log(`Fetching documentation from ${DOCS_URL}...`);
    const response = await fetch(DOCS_URL);
    if (!response.ok) throw new Error(`Failed to fetch docs: ${response.status} ${response.statusText}`);
    rawDocumentationContent = await response.text();
    console.log('Documentation fetched successfully.');
    preparedDocChunks = preprocessDocs(rawDocumentationContent);
    if (preparedDocChunks.length > 0) {
      const fuseOptions: IFuseOptions<string> = { includeScore: true, threshold: 0.5, ignoreLocation: true };
      fuseInstance = new Fuse(preparedDocChunks, fuseOptions);
      console.log(`Fuse.js initialized with ${preparedDocChunks.length} searchable chunks.`);
    } else {
      console.log('No searchable content after preprocessing. Fuse.js not initialized.');
    }
  } catch (error) {
    console.error('Error fetching or initializing documentation/Fuse.js:', error);
  }
}

async function initializeKnowledgeBase(): Promise<void> {
  try {
    console.log(`Loading knowledge base from ${KNOWLEDGE_FILE_PATH}...`);
    const fileContent = await fs.readFile(KNOWLEDGE_FILE_PATH, 'utf8');
    knowledgeBase = JSON.parse(fileContent) as KnowledgeItem[];
    if (knowledgeBase.length > 0) {
      const fuseOptions: IFuseOptions<KnowledgeItem> = {
        includeScore: true,
        threshold: 0.6,
        ignoreLocation: true,
        keys: ['question'],
      };
      knowledgeFuseInstance = new Fuse(knowledgeBase, fuseOptions);
      console.log(`Knowledge base initialized with ${knowledgeBase.length} Q&A items.`);
    } else {
      console.log('Knowledge base is empty or failed to load. Q&A search will not be available.');
    }
  } catch (error) {
    console.error('Error loading or initializing knowledge base:', error);
    knowledgeFuseInstance = null;
  }
}

async function readLastUpdateId(): Promise<number> {
  try {
    const data = await fs.readFile(LAST_UPDATE_ID_FILE, 'utf8');
    const id = parseInt(data, 10);
    return Number.isInteger(id) ? id : 0;
  } catch (error: unknown) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') return 0;
    throw error;
  }
}

async function writeLastUpdateId(id: number): Promise<void> {
  await fs.writeFile(LAST_UPDATE_ID_FILE, id.toString(), 'utf8');
}

function fixRelativeLinks(text: string, baseUrl: string): string {
  if (!text) return '';
  const linkRegex = /\[([^\]]+)\]\(([^)\s]+)\)/g;
  return text.replace(linkRegex, (match, linkText, linkUrl) => {
    if (linkUrl.startsWith('/') && !linkUrl.startsWith('//')) {
      return `[${linkText}](${baseUrl}${linkUrl})`;
    }
    return match;
  });
}

function replaceMarkdownHeadersWithBold(text: string): string {
  if (!text) return '';
  const headerRegex = /^(#{1,6})\s+(.*)/gm;
  return text.replace(headerRegex, (_match, _hashes, headerText) => {
    return `*${headerText.trim()}*`;
  });
}

function escapeForMarkdownV2(text: string): string {
  if (!text) return '';

  const linkRegex = /\[([^\]]*)\]\(([^)\s]+)\)/g;
  const links: string[] = [];
  let placeholderIndex = 0;

  // Step 1: Replace Markdown links with placeholders
  const textWithPlaceholders = text.replace(linkRegex, (match) => {
    links.push(match);
    return `__TEMP_MD_LINK_${placeholderIndex++}__`;
  });

  // Step 2: Apply original escapes (., * , -) on the text with placeholders
  let escapedText = textWithPlaceholders.replace(/\./g, '\\.');
  escapedText = escapedText.replace(/\* /g, '\\* ');
  escapedText = escapedText.replace(/-/g, '\\-');

  /*
   * Step 3: Escape standalone '[' and ']'
   * Brackets that were part of Markdown links are now safely within placeholders.
   */
  escapedText = escapedText.replace(/\[/g, '\\[');
  escapedText = escapedText.replace(/\]/g, '\\]');
  escapedText = escapedText.replace(/\(/g, '\\(');
  escapedText = escapedText.replace(/\)/g, '\\)');

  // Step 4: Restore Markdown links from placeholders
  for (let i = 0; i < links.length; i++) {
    escapedText = escapedText.replace(`__TEMP_MD_LINK_${i}__`, links[i] || '');
  }

  return escapedText;
}

bot.start((ctx) => {
  ctx.reply('Hello! I am your DevRel AI assistant. I can search our docs and try to answer your questions.');
});

bot.help((ctx) => {
  ctx.reply('Send me any message. I will search our docs for relevant info and use AI to formulate an answer.');
});

bot.on(message('text'), async (ctx) => {
  if (ctx.message.text.startsWith('/')) {
    return;
  }

  const userQuery = ctx.message.text;
  const containsQuestionMark = userQuery.includes('?');

  const isReplyToThisBot = !!(
    ctx.message.reply_to_message &&
    ctx.message.reply_to_message.from &&
    ctx.botInfo &&
    ctx.message.reply_to_message.from.id === ctx.botInfo.id
  );

  if (!containsQuestionMark && !isReplyToThisBot) {
    return;
  }

  const chatId = ctx.chat.id;
  if (!chatHistories[chatId]) chatHistories[chatId] = [];
  if (!aiClient) {
    try {
      aiClient = createOpenAIClient();
    } catch (error: unknown) {
      if (error instanceof Error) {
        console.error('Failed to create AI client:', error.message);
      } else {
        console.error('Failed to create AI client:', String(error));
      }
      ctx.reply("Sorry, I'm having trouble connecting to the AI service. Please ensure API keys are set.");
      return;
    }
  }

  let requestType: z.infer<typeof RequestTypeEnum> = 'documentation_query';
  try {
    const classificationSystemPrompt = `Your task is to classify the user\'s query into ONE of the following categories:
1. "documentation_query": The user is asking a question about Orderly Network\'s features, SDKs, APIs, technical details, how to build/use Orderly Network, or other topics likely covered in general technical documentation.
2. "bot_related_inquiry": The user is asking a meta-question about you (the bot, e.g., "who are you?", "what can you do?"), a greeting, or engaging in simple conversation related to your function as an assistant.
3. "broker_id_setup_inquiry": The user is specifically asking about setting up a Broker ID, issues related to becoming a broker, or the broker application process for Orderly Network.
4. "unrelated_query": The user's message is not a question or request *directed at you, the AI assistant*. This includes:\n    - Messages that are part of an ongoing conversation between *other users*.\n    - Direct instructions or requests *to other human users*, even if phrased with a question (e.g., "John, can you send me the file?", "Team, what's the status on X?", "Could you (the broker) please set this up for me?").\n    - General commentary not requiring an AI response.\n    - Topics completely unrelated to Orderly Network or your functions as an assistant.\n    If a message contains a question but is clearly addressed to someone other than the AI assistant, it is an "unrelated_query".

You MUST respond *only* with a single, valid JSON object. This JSON object must contain exactly one key named "requestType". The value for "requestType" MUST be one of the following exact strings: "documentation_query", "bot_related_inquiry", "broker_id_setup_inquiry", or "unrelated_query".

Example of a valid JSON response:
{"requestType": "documentation_query"}

Another example:
{"requestType": "broker_id_setup_inquiry"}

Do NOT include any other text, explanation, apologies, markdown formatting, or conversational filler in your response. Your entire response must be only the JSON object itself.`;

    const classificationMessages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: 'system', content: classificationSystemPrompt },
      ...(chatHistories[chatId] || []),
      { role: 'user', content: userQuery },
    ];

    const classificationCompletion = await aiClient.chat.completions.create({
      model: 'qwen-3-235b-a22b-instruct-2507',
      messages: classificationMessages,
      temperature: 0.1,
      response_format: { type: 'json_object' },
    });

    const rawResponse = classificationCompletion.choices[0]?.message?.content;
    if (rawResponse) {
      try {
        const jsonObject = JSON.parse(rawResponse);
        const validationResult = ClassificationResponseSchema.safeParse(jsonObject);

        if (validationResult.success) {
          requestType = validationResult.data.requestType;
        } else {
          console.error('Classification AI response failed Zod validation:', validationResult.error.flatten());
          console.error('Raw response was:', rawResponse);
        }
      } catch (jsonParseError) {
        console.error(
          'Failed to parse classification AI response string as JSON:',
          jsonParseError,
          '\nRaw response:',
          rawResponse,
        );
      }
    }
  } catch (classificationError: unknown) {
    if (classificationError instanceof Error) {
      console.error('Error during request classification AI call:', classificationError.message);
    } else {
      console.error('Error during request classification AI call:', String(classificationError));
    }
  }

  let systemPromptContent: string;
  let messagesForAnsweringAI: OpenAI.Chat.Completions.ChatCompletionMessageParam[] | undefined = undefined;
  let shouldMakeSecondAICall = true;
  let predefinedResponse = '';

  const orderlyNetworkDefinition =
    'Orderly Network is a permissionless, omnichain Central Limit Order Book (CLOB) infrastructure that unifies liquidity across multiple blockchains, delivering CEX-level performance, security, and composability with DEX transparency for seamless dApp integration.';

  const markdownInstructionsForTables = `\n- TABLES: Telegram does not render standard markdown tables (e.g., using pipes |). If you need to present tabular data, please use one of these alternatives:\n  - A series of bulleted lists (e.g., one list per conceptual row). Use a hyphen (-) followed by a space for each list item (e.g., "- First item").\n  - A pre-formatted fixed-width text block (using triple backticks \`\`\`) where you manually align columns with spaces to simulate a table.\n  - Describe the data in paragraph form if it's simple.\n  - **Do NOT use markdown pipe table syntax (e.g., | Header | Header |).**\n\n- GENERAL MARKDOWN & SPECIAL CHARACTERS: Use Markdown formatting (like *bold*, _italics_, - lists) only when necessary for readability. In all other text, do not use any special characters except for standard punctuation (e.g., '.', ',', '?', '!'). Avoid special characters like \`*\`-, \`_\` if they are not part of a Markdown formatting instruction. **If you need to use special characters like underscores (_) or hyphens (-) within words or identifiers (e.g., 'broker_id', 'some-variable'), you should enclose the entire word/identifier in single backticks (e.g., \`broker_id\`, \`some-variable\`). This will ensure they are displayed correctly and not misinterpreted by Telegram's Markdown parser.**`;

  if (requestType === 'documentation_query') {
    if (!fuseInstance) {
      ctx.reply("I'm sorry, the documentation search module isn't ready. Please try again.");
      return;
    }
    const fuseResults = fuseInstance.search(userQuery);
    let relevantParagraphsForPrompt = 'No specific documentation excerpts found for your query.';
    if (fuseResults.length > 0) {
      const topResults = fuseResults.slice(0, MAX_FUSE_RESULTS);
      let concatenatedParagraphs = '';
      let currentLength = 0;
      const separator = '\n\n---\n\n';
      for (const result of topResults) {
        const chunk = result.item;
        const lengthWithSeparator = concatenatedParagraphs.length > 0 ? separator.length : 0;
        if (currentLength + chunk.length + lengthWithSeparator <= MAX_DOC_CONTEXT_CHARACTERS) {
          if (concatenatedParagraphs.length > 0) concatenatedParagraphs += separator;
          concatenatedParagraphs += chunk;
          currentLength = concatenatedParagraphs.length;
        } else break;
      }
      if (concatenatedParagraphs.length > 0) relevantParagraphsForPrompt = concatenatedParagraphs;
    }

    let knowledgeBaseContentForPrompt = 'No relevant Q&A found in the knowledge base for your query.';
    if (knowledgeFuseInstance) {
      const knowledgeSearchResults = knowledgeFuseInstance.search(userQuery);
      if (knowledgeSearchResults.length > 0) {
        const topKnowledgeResults = knowledgeSearchResults.slice(0, MAX_KNOWLEDGE_RESULTS);
        let concatenatedKnowledge = '';
        const separator = '\n\n---\n\n';
        for (const result of topKnowledgeResults) {
          const qaPair = `Q: ${result.item.question}\nA: ${result.item.answer}`;
          const lengthWithSeparator = concatenatedKnowledge.length > 0 ? separator.length : 0;
          if (concatenatedKnowledge.length + qaPair.length + lengthWithSeparator <= MAX_KB_CONTEXT_CHARACTERS) {
            if (concatenatedKnowledge.length > 0) concatenatedKnowledge += separator;
            concatenatedKnowledge += qaPair;
          } else break;
        }
        if (concatenatedKnowledge.length > 0) knowledgeBaseContentForPrompt = concatenatedKnowledge;
      }
    }

    systemPromptContent = `${orderlyNetworkDefinition}\n\nYou are the Orderly Network Documentation Helper, an AI assistant expert in Orderly Network. Your role is to answer user questions accurately using only the Orderly Network information provided below and the conversation history.\nKey Instructions:\n1. Answer directly and concisely if the information is available in the provided text or conversation history.\n2. If the specific information needed to answer is not present in the provided text or history, clearly state that you do not have that specific information. Do not apologize.\n3. Do NOT invent answers or use any external knowledge beyond what is provided here.\n4. **CRITICAL: Absolutely do NOT mention that you are basing your answer on "provided excerpts," "documentation excerpts," "information provided," "Knowledge Base," "Q&A section," or any similar phrases referring to your source material. Simply provide the answer as the expert.**\n5. Do NOT suggest the user refer to external Orderly Network documentation, websites, support channels, or a "Q&A section" or "Knowledge Base" as if it's a separate browsable resource. You ARE the direct source for this information.\n${markdownInstructionsForTables}\n\n[Use the following Orderly Network information and conversation history to answer the user's current question]\n\nRelevant Information from Documentation:\n${relevantParagraphsForPrompt}\n\nRelevant Q&A from Knowledge Base:\n${knowledgeBaseContentForPrompt}`;
    messagesForAnsweringAI = [
      { role: 'system', content: systemPromptContent },
      ...(chatHistories[chatId] || []),
      { role: 'user', content: userQuery },
    ];
  } else if (requestType === 'bot_related_inquiry') {
    systemPromptContent = `${orderlyNetworkDefinition}\n\nYou are a friendly and helpful AI assistant for Orderly Network. The user is interacting with you directly (e.g., greeting, asking about your capabilities). Respond naturally and concisely based on the conversation history. If asked about your capabilities, state that you are the Orderly Network Documentation Helper and can answer questions about Orderly Network using its official documentation.\n${markdownInstructionsForTables}`;
    messagesForAnsweringAI = [
      { role: 'system', content: systemPromptContent },
      ...(chatHistories[chatId] || []),
      { role: 'user', content: userQuery },
    ];
  } else if (requestType === 'broker_id_setup_inquiry') {
    predefinedResponse =
      "It sounds like you're asking about Broker ID setup. This requires specific attention. I've notified @Orderly\\_Wuzhong and @Mario\\_Orderly to assist you.";
    shouldMakeSecondAICall = false;
  } else if (requestType === 'unrelated_query') {
    shouldMakeSecondAICall = false;
    predefinedResponse = '';
    return;
  } else {
    console.warn(`Unknown or unhandled requestType: '${requestType}'. Defaulting to documentation_query path.`);
    if (!fuseInstance) {
      ctx.reply("I'm sorry, the documentation search module isn't ready. Please try again.");
      return;
    }
    const fuseResults = fuseInstance.search(userQuery);
    let relevantParagraphsForPrompt = 'No specific documentation excerpts found for your query (fallback path).';
    if (fuseResults.length > 0) {
      const topResults = fuseResults.slice(0, MAX_FUSE_RESULTS);
      let concatenatedParagraphs = '';
      let currentLength = 0;
      const separator = '\n\n---\n\n';
      for (const result of topResults) {
        const chunk = result.item;
        const lengthWithSeparator = concatenatedParagraphs.length > 0 ? separator.length : 0;
        if (currentLength + chunk.length + lengthWithSeparator <= MAX_DOC_CONTEXT_CHARACTERS) {
          if (concatenatedParagraphs.length > 0) concatenatedParagraphs += separator;
          concatenatedParagraphs += chunk;
          currentLength = concatenatedParagraphs.length;
        } else break;
      }
      if (concatenatedParagraphs.length > 0) relevantParagraphsForPrompt = concatenatedParagraphs;
    }

    let knowledgeBaseContentForPrompt = 'No relevant Q&A found in the knowledge base for your query (fallback path).';
    if (knowledgeFuseInstance) {
      const knowledgeSearchResults = knowledgeFuseInstance.search(userQuery);
      if (knowledgeSearchResults.length > 0) {
        const topKnowledgeResults = knowledgeSearchResults.slice(0, MAX_KNOWLEDGE_RESULTS);
        let concatenatedKnowledge = '';
        const separator = '\n\n---\n\n';
        for (const result of topKnowledgeResults) {
          const qaPair = `Q: ${result.item.question}\nA: ${result.item.answer}`;
          const lengthWithSeparator = concatenatedKnowledge.length > 0 ? separator.length : 0;
          if (concatenatedKnowledge.length + qaPair.length + lengthWithSeparator <= MAX_KB_CONTEXT_CHARACTERS) {
            if (concatenatedKnowledge.length > 0) concatenatedKnowledge += separator;
            concatenatedKnowledge += qaPair;
          } else break;
        }
        if (concatenatedKnowledge.length > 0) knowledgeBaseContentForPrompt = concatenatedKnowledge;
      }
    }
    systemPromptContent = `${orderlyNetworkDefinition}\n\nYou are the Orderly Network Documentation Helper, an AI assistant expert in Orderly Network. Your role is to answer user questions accurately using only the Orderly Network information provided below and the conversation history.\nKey Instructions:\n1. Answer directly and concisely if the information is available in the provided text or conversation history.\n2. If the specific information needed to answer is not present in the provided text or history, clearly state that you do not have that specific information. Do not apologize.\n3. Do NOT invent answers or use any external knowledge beyond what is provided here.\n4. **CRITICAL: Absolutely do NOT mention that you are basing your answer on "provided excerpts," "documentation excerpts," "information provided," "Knowledge Base," "Q&A section," or any similar phrases referring to your source material. Simply provide the answer as the expert.**\n5. Do NOT suggest the user refer to external Orderly Network documentation, websites, support channels, or a "Q&A section" or "Knowledge Base" as if it's a separate browsable resource. You ARE the direct source for this information.\n${markdownInstructionsForTables}\n\n[Use the following Orderly Network information and conversation history to answer the user's current question]\n\nRelevant Information from Documentation:\n${relevantParagraphsForPrompt}\n\nRelevant Q&A from Knowledge Base:\n${knowledgeBaseContentForPrompt}`;
    messagesForAnsweringAI = [
      { role: 'system', content: systemPromptContent },
      ...(chatHistories[chatId] || []),
      { role: 'user', content: userQuery },
    ];
  }

  if (shouldMakeSecondAICall && messagesForAnsweringAI) {
    try {
      const chatCompletion = await aiClient.chat.completions.create({
        model: 'qwen-3-235b-a22b-instruct-2507',
        messages: messagesForAnsweringAI,
      });
      const aiResponse = chatCompletion.choices[0]?.message?.content;

      if (aiResponse) {
        let processedAiResponse = fixRelativeLinks(aiResponse, DOCS_BASE_URL);
        processedAiResponse = replaceMarkdownHeadersWithBold(processedAiResponse);
        const telegramSafeResponse = escapeForMarkdownV2(processedAiResponse);

        ctx.reply(telegramSafeResponse, { parse_mode: 'MarkdownV2' });

        chatHistories[chatId].push({ role: 'user', content: userQuery });
        chatHistories[chatId].push({ role: 'assistant', content: processedAiResponse });
        if (chatHistories[chatId].length > MAX_HISTORY_MESSAGES) {
          chatHistories[chatId] = chatHistories[chatId].slice(-MAX_HISTORY_MESSAGES);
        }
      } else {
        ctx.reply('Sorry, I received an empty response from the AI.');
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } catch (error: any) {
      console.error('Error calling AI service for answering:', error.message);
      if (error.response && error.response.data && error.response.data.error) {
        console.error('AI API Error details:', JSON.stringify(error.response.data.error));
        ctx.reply(`Sorry, there was an error with the AI service: ${error.response.data.error.message}`);
      } else {
        ctx.reply('Sorry, I encountered an error while trying to get an answer from the AI.');
      }
    }
  } else if (predefinedResponse) {
    let finalPredefinedResponse = replaceMarkdownHeadersWithBold(predefinedResponse);
    finalPredefinedResponse = escapeForMarkdownV2(finalPredefinedResponse);
    ctx.reply(finalPredefinedResponse, { parse_mode: 'MarkdownV2' });

    chatHistories[chatId].push({ role: 'user', content: userQuery });
    chatHistories[chatId].push({ role: 'assistant', content: predefinedResponse });
    if (chatHistories[chatId].length > MAX_HISTORY_MESSAGES) {
      chatHistories[chatId] = chatHistories[chatId].slice(-MAX_HISTORY_MESSAGES);
    }
  }
});

async function pollUpdates() {
  let lastUpdateId = await readLastUpdateId();
  console.log(`Bot starting... Initial last processed update ID: ${lastUpdateId}`);

  while (true) {
    try {
      const updates = await bot.telegram.getUpdates(5, 100, lastUpdateId + 1, ['message']);

      if (updates.length > 0) {
        for (const update of updates) {
          bot.handleUpdate(update);
          lastUpdateId = update.update_id;
        }
        await writeLastUpdateId(lastUpdateId);
        console.log(`Successfully processed updates up to ID: ${lastUpdateId}`);
      }
    } catch (error) {
      console.error('Error fetching or processing updates:', error);
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
  }
}

console.log('Initializing bot with manual update polling...');

async function main() {
  await initializeDocumentation();
  await initializeKnowledgeBase();
  pollUpdates().catch((error) => {
    console.error('Critical error in polling loop. Exiting.', error);
    process.exit(1);
  });
}

main();

process.once('SIGINT', () => {
  console.log('SIGINT received, stopping bot...');
  bot.stop('SIGINT');
  process.exit(0);
});

process.once('SIGTERM', () => {
  console.log('SIGTERM received, stopping bot...');
  bot.stop('SIGTERM');
  process.exit(0);
});
