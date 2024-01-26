import OpenAI from 'openai';

import dotenv from 'dotenv';
dotenv.config({ path: './.env' });


// You still have to pass a key because the client requires it, but it doesn't have to be valid since we're using a third party LLM
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ASTRA_DB_APPLICATION_TOKEN = process.env.ASTRA_DB_APPLICATION_TOKEN;

const baseUrl = process.env.base_url || "https://open-assistant-ai.astra.datastax.com/v1";

const openai = new OpenAI({
    base_url: baseUrl,
    api_key: OPENAI_API_KEY,
    default_headers: {
        "astra-api-token": ASTRA_DB_APPLICATION_TOKEN,
    },
});


async function main() {
  const stream = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: "what's your favorite ice cream flavor?" }],
    stream: true,
  });
  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}

main();
