import OpenAI from 'openai';

const BASE_URL = process.env.OPENAI_BASE_URL || 'https://open-assistant-ai.astra.datastax.com';

const customFetch = async (url, init) => {
  const urlObj = new URL(url);
  const modifiedUrl = `${BASE_URL}${urlObj.pathname}${urlObj.search}`;

  const modifiedInit = {
    ...init,
    headers: {
      ...init.headers,
      //Authorization: `Bearer YOUR_API_KEY`,
    },
  };

  console.log("modified url is: " + modifiedUrl)
  const response = await fetch(modifiedUrl, modifiedInit);

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`Request failed: ${response.status} ${response.statusText} - ${errorBody}`);
  }

  return response;
};

const client = new OpenAI({
  apiKey: process.env['OPENAI_API_KEY'], // This is the default and can be omitted
  fetch: customFetch,
});

async function main() {
  const result = await client.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: 'Say hello!' }],
  });

  console.log(result.choices[0].message.content);
}

main().catch(console.error);
