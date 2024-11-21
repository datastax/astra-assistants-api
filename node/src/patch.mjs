import fetch from 'node-fetch';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

const BASE_URL = process.env.BASE_URL || 'https://open-assistant-ai.astra.datastax.com';
const LITELLM_MODELS_URL = 'https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json';
const ASTRA_DB_APPLICATION_TOKEN = process.env.ASTRA_DB_APPLICATION_TOKEN;

// Map of API keys for each provider
const providerApiKeys = {
  openai: process.env.OPENAI_API_KEY,
  anthropic: process.env.ANTHROPIC_API_KEY,
  groq: process.env.GROQ_API_KEY,
  gemini: process.env.GEMINI_API_KEY,
  perplexity: process.env.PERPLEXITY_API_KEY,
  cohere: process.env.COHERE_API_KEY,
};

// Fetch model metadata
async function fetchModelData() {
  try {
    const response = await fetch(LITELLM_MODELS_URL);
    if (!response.ok) {
      throw new Error(`Failed to fetch model metadata: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching model data:', error.message);
    throw error;
  }
}

async function assignHeadersForModel(body) {
  let headers = {};
  let model = body.model || null;
  let assistantId = body.assistant_id || null;
  let embeddingModel = null;
  let apiKey = null;

  // If assistant_id is provided, fetch its details
  if (!model && assistantId) {
    try {
      const assistantResponse = await fetch(`${BASE_URL}/assistants/${assistantId}`);
      const assistantData = await assistantResponse.json();
      model = assistantData.model || null;

      // Check for vector store resources to resolve embedding model
      const vectorStoreId = assistantData.tool_resources?.file_search?.vector_store_ids?.[0];
      if (vectorStoreId) {
        const vectorStoreResponse = await fetch(`${BASE_URL}/vectorStores/${vectorStoreId}`);
        const vectorStoreData = await vectorStoreResponse.json();

        // Use the first file's embedding model, if available
        const fileId = vectorStoreData.files?.[0]?.id;
        if (fileId) {
          const fileResponse = await fetch(`${BASE_URL}/files/${fileId}`);
          const fileData = await fileResponse.json();
          embeddingModel = fileData.model|| null;
        }
      }
    } catch (error) {
      console.error('Error fetching model or embedding model:', error);
    }
  }

  // Determine the provider and corresponding API key
  if (model || embeddingModel) {
    try {
      const provider = model ? getProviderFromModel(model) : getProviderFromModel(embeddingModel);
      apiKey = providerApiKeys[provider];
      if (!apiKey) {
        throw new Error(`No API key found for provider: ${provider}`);
      }

      headers['authorization'] = `Bearer ${apiKey}`;
    } catch (error) {
      console.error('Error determining API key for model:', error);
    }
  }

  // Add model-related headers
  if (model) {
    headers['X-Model-Used'] = model;
  }
  if (embeddingModel) {
    headers['X-Embedding-Model-Used'] = embeddingModel;
  }

  return headers;
}

function getProviderFromModel(model) {
  // Logic for determining the provider based on model name
  if (model.startsWith('gpt')) return 'openai';
  if (model.startsWith('claude')) return 'anthropic';
  if (model.startsWith('gemini')) return 'gemini';
  if (model.startsWith('cohere')) return 'cohere';
  // Add additional logic for other providers
  throw new Error(`Unknown provider for model: ${model}`);
}


// Replace the fetch function in the existing OpenAI client
function patch(existingClient) {
  // Save the existing fetch function (if needed for fallback or reference)
  const originalFetch = existingClient.fetch;

  const customFetch = async (url, init) => {
    const urlObj = new URL(url, BASE_URL);
    const modifiedUrl = `${BASE_URL}${urlObj.pathname}${urlObj.search}`;
  
    // Parse the body to pass to assignHeadersForModel
    let parsedBody = {};
    if (init.body && typeof init.body === 'string') {
      try {
        parsedBody = JSON.parse(init.body);
      } catch (error) {
        console.error('Error parsing request body:', error);
      }
    }
  
    // Resolve dynamic headers
    const dynamicHeaders = await assignHeadersForModel(parsedBody);

    // Combine dynamic headers with existing headers
    const customHeaders = {
      ...init.headers,
      ...dynamicHeaders,
      'astra-api-token': ASTRA_DB_APPLICATION_TOKEN,
    };
  
    const modifiedInit = {
      ...init, // Preserve existing fetch options
      headers: customHeaders, // Use modified headers
    };
  
    const response = await fetch(modifiedUrl, modifiedInit);
  
    if (!response.ok) {
      const errorBody = await response.text();
      console.error(`Response Error: ${response.status} ${response.statusText}`);
      console.error('Response Headers:', JSON.stringify(response.headers.raw(), null, 2));
      console.error('Response Body:', errorBody);
      throw new Error(`Request failed: ${response.status} ${response.statusText} - ${errorBody}`);
     }
  
    return response;
  };

  // Replace the fetch function in the existing client
  existingClient.fetch = customFetch;

  // Return the modified client
  return existingClient;
}

export {
  patch,
  assignHeadersForModel,
  fetchModelData,
};
