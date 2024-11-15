import { patch } from 'astra-assistants'; // Replace with your local package if not published

// Mock an existing OpenAI client (replace with a real client in production)
import OpenAI from 'openai';

const existingClient = new OpenAI({
  apiKey: 'your-openai-api-key', // Replace with your actual OpenAI API key
});

// Apply the patch
const patchedClient = patch(existingClient);

// Example usage: Call the OpenAI API with the patched client
async function runExample() {
  try {
    const response = await patchedClient.chat.completions.create({
      model: 'gpt-3.5-turbo',
      messages: [{ role: 'user', content: 'Say hello!' }],
    });

    console.log('Response:', response);
  } catch (error) {
    console.error('Error:', error);
  }
}

// Run the example
runExample();
