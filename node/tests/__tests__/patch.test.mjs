import { patch } from '../../src/index.mjs';
import OpenAI from "openai";

describe('patch', () => {
  it('should construct the correct URL and headers', async () => {
      try {
        // Create a base OpenAI client
        let client = new OpenAI({
          apiKey: process.env.OPENAI_API_KEY,
        });

        // Replace fetch in the existing client
        let new_client = patch(client);

        // Make a request using the enhanced client
        const result = await new_client.chat.completions.create({
          model: 'claude-3-haiku-20240307',
          messages: [{ role: 'user', content: 'Hello, world!' }],
        });

        console.log('Result:', result);
      } catch (error) {
        console.error('Error:', error.message);
        throw error
      }
  })
})
