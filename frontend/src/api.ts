export type Source = {
  title?: string | null;
  matched_question?: string | null;
  url?: string | null;
  page?: number | null;
  verification_status?: string | null;
  verification_note?: string | null;
  source_page_label?: string | null;
  category?: string | null;
  document_id?: string | null;
  source_file?: string | null;
};

export type CollectionRecord = {
  document_id: string;
  category: string;
  title: string;
  source_file: string;
  answer: string;
  original_record: Record<string, unknown>;
};

export type ChatResult = {
  answer: string;
  sources: Source[];
  records: CollectionRecord[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
  intent: string;
  category?: string | null;
  status: string;
  query: string;
  retrieved_document_ids: string[];
  retrieval_method: string;
  score?: number | null;
  follow_up_questions: string[];
  mode_label: string;
  generation_status: string;
};

export type EvaluationMetric = {
  status: string;
  top1_accuracy?: number;
  top3_accuracy?: number;
  mrr?: number;
  answerable_acceptance_rate?: number;
  rejection_accuracy?: number;
  answerable_queries?: number;
  rejection_queries?: number;
  false_acceptances?: number;
};

export type EvaluationResult = {
  status: string;
  message?: string;
  dataset?: string;
  knowledge_documents?: number;
  answerable_query_count?: number;
  rejection_query_count?: number;
  semantic_model?: string | null;
  semantic_model_available?: boolean;
  metrics?: Record<string, EvaluationMetric>;
  limitations?: string[];
};

const API_URL = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, '')
  || (import.meta.env.MODE === 'development'
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : '');

const REQUEST_TIMEOUT_MS = 45_000;

export class ApiRequestError extends Error {
  constructor(message: string, readonly kind: 'network' | 'timeout' | 'http' | 'response') {
    super(message);
    this.name = 'ApiRequestError';
  }
}

function requestUrl(path: string): string {
  const normalizedPath = path.replace(/^\/+/, '');
  return `${API_URL}/${normalizedPath}`;
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  if (!API_URL) {
    throw new Error('Backend URL is not configured. Set VITE_API_BASE_URL for this deployment.');
  }

  const controller = new AbortController();
  const externalSignal = init?.signal;
  const abortFromExternalSignal = () => controller.abort();
  if (externalSignal?.aborted) controller.abort();
  else externalSignal?.addEventListener('abort', abortFromExternalSignal, { once: true });
  const timeoutId = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    let response: Response;
    try {
      response = await fetch(requestUrl(path), {
        ...init,
        signal: controller.signal,
      });
    } catch {
      if (controller.signal.aborted && !externalSignal?.aborted) {
        throw new ApiRequestError('The backend took too long to respond. Please try again.', 'timeout');
      }
      throw new ApiRequestError(
        'Could not connect to CampusConnect AI. Check the backend connection and try again.',
        'network',
      );
    }

    if (!response.ok) {
      const body = await response.json().catch(() => null) as { detail?: string } | null;
      throw new ApiRequestError(
        body?.detail || `The backend request failed (${response.status}). Please try again.`,
        'http',
      );
    }

    try {
      return await response.json() as T;
    } catch {
      throw new ApiRequestError('The backend returned an unreadable response. Please try again.', 'response');
    }
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener('abort', abortFromExternalSignal);
  }
}

export async function sendChatRequest(
  request: {
    question: string;
    mode: 'tfidf' | 'semantic' | 'hybrid';
    page: number;
    page_size: number;
    history?: string[];
  },
): Promise<ChatResult> {
  const result = await fetchJson<unknown>('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!isChatResult(result)) {
    throw new ApiRequestError(
      'The backend returned an unexpected chat response. Please try again.',
      'response',
    );
  }
  return result;
}

function isChatResult(value: unknown): value is ChatResult {
  if (!value || typeof value !== 'object') return false;
  const result = value as Partial<ChatResult>;
  return typeof result.answer === 'string'
    && typeof result.status === 'string'
    && typeof result.intent === 'string'
    && Array.isArray(result.sources)
    && Array.isArray(result.records)
    && Array.isArray(result.retrieved_document_ids);
}
