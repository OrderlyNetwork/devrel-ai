import tseslint from 'typescript-eslint';
import eslintPluginPrettierRecommended from 'eslint-plugin-prettier/recommended';

export default [
  {
    languageOptions: {
      ecmaVersion: 2020,
      sourceType: 'module',
    },
  },

  ...tseslint.configs.recommended,

  eslintPluginPrettierRecommended,

  {
    rules: {
      'no-inline-comments': 'error',
      'multiline-comment-style': ['error', 'starred-block'],
    },
  },
];
