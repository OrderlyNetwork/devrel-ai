/*
 * analyze_chat_openai.js
 *
 * This script analyzes a single chat export JSON file using the OpenAI API
 * to extract DevRel-related questions and answers about Orderly Network.
 *
 * Prerequisites:
 * 1. Node.js installed.
 * 2. OpenAI API key set in a .env file (OPENAI_API_KEY=your_api_key_here).
 * 3. npm install openai dotenv
 *
 * Usage:
 * node analyze_chat_openai.js <path_to_chat_file.json>
 * Example:
 * node analyze_chat_openai.js telegram_chat_exports/chat_Orderly_DevRel.json
 *
 * The script will create an analysis file (e.g., chat_Orderly_DevRel_analysis.json)
 * in the same directory as the input chat file.
 */

import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';
import dotenv from 'dotenv';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const OPENAI_MODEL = 'o4-mini';
// const OPENAI_MODEL = 'gpt-4.1-nano';
const CHAT_EXPORTS_DIR = 'telegram_chat_exports';
const FINAL_ANALYSIS_FILENAME = 'final_orderly_qa_analysis.json';
// Set to null or Infinity to process all, or a number for testing
const MAX_FILES_TO_PROCESS = null;

function formatChatMessages(messages) {
  if (!Array.isArray(messages)) {
    return '';
  }
  return messages
    .map((message) => {
      const from = message.from || message.from_id || 'Unknown User';
      const date = message.date || '';
      let textContent = '';

      if (typeof message.text === 'string') {
        textContent = message.text;
      } else if (Array.isArray(message.text)) {
        textContent = message.text.map((part) => (typeof part === 'string' ? part : part.text || '')).join('');
      }
      if (textContent.trim() === '') {
        return null;
      }
      return `${from} (${date}): ${textContent}`;
    })
    .filter(Boolean)
    .join('\n');
}

async function getOpenAIAnalysis(chatContent, existingQAPairs = []) {
  if (!chatContent || chatContent.trim() === '') {
    console.log('Chat content is empty. Skipping OpenAI analysis.');
    return [];
  }

  let systemPrompt =
    'You are an AI assistant specialized in analyzing chat transcripts for Developer Relations insights.\n';
  systemPrompt +=
    'Your primary task is to identify commonly asked Developer Relations questions about Orderly Network that appear in the provided transcript and extract their corresponding answers directly and accurately from the chat.\n';
  systemPrompt +=
    'Focus on technical questions, integration help, API usage, SDKs, product features, and similar topics a developer might ask.\n\n';
  systemPrompt += 'CRITICAL INSTRUCTIONS FOR ANSWER QUALITY:\n';
  systemPrompt +=
    "1. ACCURACY: The answer MUST directly and precisely address the specific question asked. Pay extremely close attention to keywords in the question (e.g., differentiate between 'withdrawal' and 'deposit', specific feature names, etc.) and ensure your answer corresponds exactly to what was asked. Do not provide information about related but distinct topics.\n";
  systemPrompt +=
    '2. ACTIONABILITY: The \"answer\" field MUST be actionable for a developer. It should provide clear guidance, next steps, or specific pointers to where information can be found.\n';
  systemPrompt +=
    '3. DETAIL FROM CHAT: If an API endpoint, SDK function, or specific technical process is mentioned, the answer MUST attempt to include available details from the chat, such as:\n';
  systemPrompt +=
    '    *   LOCATION OF INFORMATION: If the chat mentions where a developer can find more details (e.g., \"in the API docs under \'User Management\'\", \"in the SDK\'s `examples/` directory\", or a specific section of a guide), include this.\n';
  systemPrompt +=
    '    *   KEY PARAMETERS/USAGE: If crucial parameters, brief usage patterns, or relevant small code snippets are in the chat, include them.\n';
  systemPrompt +=
    '4. HANDLING LINKS: When the original chat message contains a hyperlink (e.g., to documentation like `[link text](URL)` or a raw URL):\n';
  systemPrompt +=
    '    *   If the link is to specific, highly relevant technical documentation (e.g., a deep link to an API endpoint page or SDK function details) that directly clarifies how to use a mentioned feature, you MAY include this single URL in your answer if it is the most effective way to make the answer actionable and no textual summary is available in the chat.\n';
  systemPrompt +=
    "    *   Otherwise, do NOT include the URL or markdown link. Instead, describe where the developer can find the information textually (e.g., 'Refer to the official Orderly Network API documentation, under the Trading Endpoints section') or summarize the key information if the chat text itself provides that summary.\n";
  systemPrompt +=
    '5. INCOMPLETE INFORMATION: If the chat confirms a feature/topic but provides NO actionable details (like documentation location, API/SDK specifics), the answer should state this explicitly (e.g., \"The chat confirms X feature exists, but specific implementation details or documentation pointers were not provided in this segment.\").\n\n';
  systemPrompt += 'PREVIOUSLY EXTRACTED Q&A PAIRS (for context, avoid duplicates, refine if possible):\n';
  systemPrompt += JSON.stringify(existingQAPairs, null, 2) + '\n\n';
  systemPrompt +=
    'Based on the CURRENT CHAT TRANSCRIPT below, extract NEW questions and answers, or provide REFINED answers for the existing ones if the new context is significantly better or more accurate.\n';
  systemPrompt +=
    'When doing so, pay attention to the `last_referenced_date` of existing Q&A pairs. If the CURRENT CHAT TRANSCRIPT contains newer information (i.e., messages with a later date) that contradicts or significantly updates an existing answer, you MUST prioritize the newer information and update the `last_referenced_date` accordingly.\n\n';
  systemPrompt +=
    'Format your output ONLY as a valid JSON object with a single key "qa_pairs". The value of this "qa_pairs" key must be a JSON array of objects.\n';
  systemPrompt +=
    'Each object in the array must have three string fields: "question", "answer", and "last_referenced_date".\n';
  systemPrompt +=
    'The "last_referenced_date" should be the date (e.g., "YYYY-MM-DDTHH:MM:SS" from the chat message) of the latest message in the CURRENT CHAT TRANSCRIPT that was used to formulate or confirm the answer.\n';
  systemPrompt +=
    'The "question" should be formulated as precisely as possible to capture the full context and nuance of the developer\'s query, even if the question needs to be somewhat lengthy to do so. It should accurately reflect what a developer specifically asked.\n\n';
  systemPrompt += 'Example of the required JSON output format for NEW or REFINED Q&A from the CURRENT transcript:\n';
  systemPrompt += '{\n';
  systemPrompt += '  "qa_pairs": [\n';
  systemPrompt += '    {\n';
  systemPrompt +=
    '      "question": "How do I set up the Orderly SDK for Python, including initial authentication?",\n';
  systemPrompt +=
    "      \"answer\": \"To set up the Orderly SDK for Python, first install it using 'pip install orderly-sdk'. Then, initialize the client with your API key, secret, and user ID. Key methods for placing orders are part of the trading module. For detailed setup and authentication examples, refer to the SDK's official README on the Orderly Network GitHub repository or the 'SDK Authentication' section of the developer portal.\",\n";
  systemPrompt += '      "last_referenced_date": "2023-11-15T14:30:00"\n';
  systemPrompt += '    },\n';
  systemPrompt += '    {\n';
  systemPrompt +=
    '      "question": "What are the specific API endpoints for fetching a user\'s complete trade history and their currently open orders under a specific broker ID?",\n';
  systemPrompt +=
    '      "answer": "To fetch a user\'s complete trade history, use the `GET /v1/history/trades` endpoint. For their current open orders, use `GET /v1/orders`. Both require the `orderly_account_id` and are detailed in the \'Private Endpoints\' section of the Orderly Network API documentation. Ensure you handle pagination for trade history if many trades exist.",\n';
  systemPrompt += '      "last_referenced_date": "2023-11-10T09:00:00"\n';
  systemPrompt += '    },\n';
  systemPrompt += '    {\n';
  systemPrompt +=
    '      "question": "How can I implement take profit and stop loss orders for an existing position using the latest SDK version?",\n';
  systemPrompt +=
    '      "answer": "The latest SDK version supports positional Take Profit/Stop Loss (TP/SL) orders. You should look for functions like `create_positional_tpsl_order` or similar within the SDK\'s trading or positions module. These functions typically require the position ID, trigger price, and order quantity. For the exact function names and parameters, consult the SDK\'s API reference documentation or recent changelogs.",\n';
  systemPrompt += '      "last_referenced_date": "2024-04-25T17:10:11"\n';
  systemPrompt += '    }\n';
  systemPrompt += '  ]\n';
  systemPrompt += '}\n\n';
  systemPrompt +=
    'If no new or significantly refined DevRel questions and answers are found in the current transcript, return: { "qa_pairs": [] }\n';
  systemPrompt +=
    'The entire response MUST be ONLY this JSON object. No other text or explanations before or after the JSON.';

  let userPrompt = 'CURRENT CHAT TRANSCRIPT related to Orderly Network:\n\n';
  userPrompt += '--- CURRENT CHAT TRANSCRIPT ---\n';
  userPrompt += chatContent + '\n';
  userPrompt += '--- END CURRENT CHAT TRANSCRIPT ---\n\n';
  userPrompt +=
    'Please analyze THIS CURRENT transcript, considering the previously extracted Q&A, and provide the output in the specified JSON object format.';

  try {
    const completion = await openai.chat.completions.create({
      model: OPENAI_MODEL,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      // temperature: 0.2,
      response_format: { type: 'json_object' },
    });

    const responseContent = completion.choices[0]?.message?.content;
    if (!responseContent) {
      console.error('OpenAI returned an empty response.');
      return [];
    }

    console.log('\n--- OpenAI Raw Response Snippet ---');
    console.log(responseContent.substring(0, 200) + (responseContent.length > 200 ? '...' : ''));
    console.log('--- End OpenAI Raw Response Snippet ---\n');

    try {
      const parsedResponse = JSON.parse(responseContent);
      if (parsedResponse && parsedResponse.qa_pairs && Array.isArray(parsedResponse.qa_pairs)) {
        return parsedResponse.qa_pairs;
      } else {
        console.error(
          'OpenAI response was not in the expected format: { "qa_pairs": [...] }. Response:',
          responseContent,
        );
        return [];
      }
    } catch (e) {
      console.error('Error parsing JSON response from OpenAI:', e);
      console.error('Raw OpenAI response that failed parsing:', responseContent);
      return [];
    }
  } catch (error) {
    console.error('Error calling OpenAI API:', error.message);
    if (error.response && error.response.data) {
      console.error('OpenAI API Error Details:', JSON.stringify(error.response.data, null, 2));
    } else if (error.status && error.error) {
      console.error('OpenAI API Error Status:', error.status);
      console.error('OpenAI API Error Details:', JSON.stringify(error.error, null, 2));
    }
    return [];
  }
}

async function main() {
  console.log(
    `Starting analysis. Max files to process: ${MAX_FILES_TO_PROCESS === null ? 'All' : MAX_FILES_TO_PROCESS}`,
  );
  let allChatFiles = [];
  try {
    const files = fs.readdirSync(CHAT_EXPORTS_DIR);
    allChatFiles = files
      .filter((file) => file.startsWith('chat_') && file.endsWith('.json') && !file.endsWith('_analysis.json'))
      .map((file) => {
        const filePath = path.join(CHAT_EXPORTS_DIR, file);
        try {
          const stats = fs.statSync(filePath);
          return { name: file, path: filePath, size: stats.size };
        } catch (e) {
          console.error(`Could not get stats for file ${filePath}: ${e.message}`);
          return null;
        }
      })
      .filter((fileObj) => fileObj !== null);

    allChatFiles.sort((a, b) => b.size - a.size);
    console.log(`Found ${allChatFiles.length} chat files to analyze.`);
  } catch (e) {
    console.error(`Error reading chat export directory ${CHAT_EXPORTS_DIR}: ${e.message}`);
    process.exit(1);
  }

  if (allChatFiles.length === 0) {
    console.log('No chat files found to analyze.');
    return;
  }

  const filesToProcess = MAX_FILES_TO_PROCESS === null ? allChatFiles : allChatFiles.slice(0, MAX_FILES_TO_PROCESS);
  console.log(`Processing ${filesToProcess.length} files (sorted by size, largest first).`);

  let cumulativeQAPairs = [];
  let filesProcessedCount = 0;

  for (const fileInfo of filesToProcess) {
    filesProcessedCount++;
    console.log(
      `\n--- Processing file ${filesProcessedCount}/${filesToProcess.length}: ${fileInfo.name} (${(fileInfo.size / (1024 * 1024)).toFixed(2)} MB) ---`,
    );

    let chatData;
    try {
      const fileContent = fs.readFileSync(fileInfo.path, 'utf8');
      chatData = JSON.parse(fileContent);
    } catch (e) {
      console.error(`Error reading or parsing chat file ${fileInfo.path}:`, e.message);
      continue;
    }

    const messages = chatData.messages;
    if (!messages) {
      console.error(`Error: Chat file ${fileInfo.name} does not contain a "messages" array.`);
      continue;
    }

    const formattedChat = formatChatMessages(messages);

    if (formattedChat.trim() === '') {
      console.log(`No text content found in messages for ${fileInfo.name}. Skipping OpenAI call.`);
      continue;
    }

    console.log(`Formatted chat content length for ${fileInfo.name}: ${formattedChat.length} characters.`);
    const MAX_CHARS_APPROX = 100000 * 3;
    if (formattedChat.length > MAX_CHARS_APPROX) {
      console.warn(
        `Warning: Formatted chat content for ${fileInfo.name} is very long (${formattedChat.length} chars) and might exceed model context limits or be too costly. Proceeding with caution.`,
      );
    }

    console.log(
      `Sending content of ${fileInfo.name} to OpenAI for analysis (cumulative Q&A: ${cumulativeQAPairs.length})...`,
    );
    const newQAPairs = await getOpenAIAnalysis(formattedChat, cumulativeQAPairs);

    if (newQAPairs.length > 0) {
      console.log(`OpenAI analysis for ${fileInfo.name} returned ${newQAPairs.length} new/refined Q/A pairs.`);
      cumulativeQAPairs = cumulativeQAPairs.concat(newQAPairs);
      console.log(`Cumulative Q&A pairs: ${cumulativeQAPairs.length}`);
    } else {
      console.log(`OpenAI analysis for ${fileInfo.name} returned no new/refined Q/A pairs or an error occurred.`);
    }
  }

  const finalAnalysisFilePath = path.join(CHAT_EXPORTS_DIR, FINAL_ANALYSIS_FILENAME);
  try {
    fs.writeFileSync(finalAnalysisFilePath, JSON.stringify(cumulativeQAPairs, null, 2));
    console.log(`\n--- Final analysis complete ---`);
    console.log(`Successfully saved ${cumulativeQAPairs.length} Q/A pairs to ${finalAnalysisFilePath}`);
  } catch (e) {
    console.error(`Error writing final analysis file ${finalAnalysisFilePath}:`, e.message);
  }
}

main().catch((err) => {
  console.error('Unhandled error in main function:', err);
  process.exit(1);
});
