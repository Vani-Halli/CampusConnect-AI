import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react';
import {
  ArrowDown,
  ArrowUp,
  Activity,
  BookOpen,
  Bot,
  CalendarDays,
  Check,
  Clipboard,
  Clock3,
  ExternalLink,
  GraduationCap,
  History,
  Menu,
  MessageCircle,
  Plus,
  Search,
  Send,
  Sparkles,
  Trash2,
  X,
} from 'lucide-react';
import {
  fetchJson,
  sendChatRequest,
  type ChatResult,
  type CollectionRecord,
  type EvaluationResult,
  type Source,
} from './api';

type Message = {
  id: number;
  role: 'assistant' | 'user';
  text: string;
  sources?: Source[];
  question?: string;
  result?: ChatResult;
  timestamp: number;
};

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

type KnowledgeRecord = {
  document_id: string;
  category: string;
  title: string;
  source_file: string;
  original_record: Record<string, unknown>;
};

type KnowledgeResponse = {
  items: KnowledgeRecord[];
  total: number;
  page: number;
  page_size: number;
  has_more: boolean;
};

type View = 'chat' | 'knowledge' | 'categories' | 'evaluation' | 'about';

const PAGE_SIZE = 12;
const CHAT_HISTORY_KEY = 'campusconnect-ai-chat-history-v1';
const WELCOME_MESSAGE = 'Hello! Ask me about Basaveshwar Engineering College (BEC), Bagalkote. Answers are based on the current BEC knowledge base and have not been independently verified.';
const SUGGESTED_QUESTIONS = [
  'What courses are offered?',
  'What are the admission requirements?',
  'What is the fee structure?',
  'What placement opportunities are available?',
  'Are internships available?',
];
const CATEGORY_QUESTIONS: Record<string, string> = {
  Admissions: 'What is the full name and location of BEC?',
  Courses: 'Which B.E. disciplines are offered in the aided section?',
  Fees: 'What are the B.E. fee components for 2026-27?',
  Placements: 'How many students were eligible for placement in the 2025 batch at BEC?',
  'Student Life': 'Does BEC provide hostel facilities for students?',
  Examinations: 'When did Basaveshwar Engineering College become autonomous?',
  'Academic Verification': 'Who can request academic document verification from BEC?',
  Administration: 'What is the parent trust of Basaveshwar Engineering College?',
  'College Contacts': 'What is the general contact information for BEC?',
  Scholarships: 'Which scholarships are listed by Basaveshwar Engineering College?',
  'Student Support': 'What student support information is available at BEC?',
};
const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  Admissions: 'Find BEC admission and application information.',
  Courses: 'Search BEC programme and course information.',
  Fees: 'Review fee details supplied in the dataset; confirm current amounts with BEC.',
  Placements: 'Search placement information and supplied batch figures.',
  'Student Life': 'Browse hostel and student-life information.',
  Examinations: 'Search examinations and academic information.',
  'Academic Verification': 'Find academic document verification information.',
  Administration: 'Browse college administration and governance information.',
  'College Contacts': 'Find contact information supplied for BEC.',
  Scholarships: 'Search scholarship information.',
  'Student Support': 'Browse student-support records.',
};

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(displayValue).filter(Boolean).join(', ');
  if (value && typeof value === 'object') {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${key.replace(/_/g, ' ')}: ${displayValue(item)}`)
      .filter(Boolean)
      .join(' · ');
  }
  return value === null || value === undefined ? '' : String(value);
}

function createWelcomeMessage(): Message {
  return {
    id: 0,
    role: 'assistant',
    text: WELCOME_MESSAGE,
    timestamp: Date.now(),
  };
}

function createConversationId(): string {
  try {
    return window.crypto.randomUUID();
  } catch {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function isTimestamp(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && !Number.isNaN(new Date(value).getTime());
}

function isStoredSource(value: unknown): value is Source {
  if (!isObject(value)) return false;
  const stringFields = [
    'title',
    'matched_question',
    'url',
    'verification_status',
    'verification_note',
    'source_page_label',
    'category',
    'document_id',
    'source_file',
  ];
  return stringFields.every((field) => value[field] === undefined
      || value[field] === null
      || typeof value[field] === 'string')
    && (value.page === undefined || value.page === null || typeof value.page === 'number');
}

function isStoredCollectionRecord(value: unknown): value is CollectionRecord {
  return isObject(value)
    && typeof value.document_id === 'string'
    && typeof value.category === 'string'
    && typeof value.title === 'string'
    && typeof value.source_file === 'string'
    && typeof value.answer === 'string'
    && isObject(value.original_record);
}

function isStoredChatResult(value: unknown): value is ChatResult {
  return isObject(value)
    && typeof value.answer === 'string'
    && Array.isArray(value.sources)
    && value.sources.every(isStoredSource)
    && Array.isArray(value.records)
    && value.records.every(isStoredCollectionRecord)
    && typeof value.total === 'number'
    && typeof value.page === 'number'
    && typeof value.page_size === 'number'
    && typeof value.has_more === 'boolean'
    && typeof value.intent === 'string'
    && typeof value.status === 'string'
    && typeof value.query === 'string'
    && Array.isArray(value.retrieved_document_ids)
    && value.retrieved_document_ids.every((id) => typeof id === 'string')
    && (value.category === undefined || value.category === null || typeof value.category === 'string')
    && typeof value.retrieval_method === 'string'
    && (value.score === undefined || value.score === null || typeof value.score === 'number')
    && Array.isArray(value.follow_up_questions)
    && value.follow_up_questions.every((question) => typeof question === 'string')
    && typeof value.mode_label === 'string'
    && typeof value.generation_status === 'string';
}

function isStoredMessage(value: unknown): value is Message {
  if (!isObject(value)
    || typeof value.id !== 'number'
    || !Number.isFinite(value.id)
    || (value.role !== 'assistant' && value.role !== 'user')
    || typeof value.text !== 'string'
    || !isTimestamp(value.timestamp)) {
    return false;
  }

  if (value.sources !== undefined
    && (!Array.isArray(value.sources) || !value.sources.every(isStoredSource))) {
    return false;
  }
  if (value.question !== undefined && typeof value.question !== 'string') return false;
  if (value.result !== undefined && !isStoredChatResult(value.result)) return false;
  return true;
}

function loadConversations(): Conversation[] {
  try {
    const stored = window.localStorage.getItem(CHAT_HISTORY_KEY);
    if (!stored) return [];
    const parsed: unknown = JSON.parse(stored);
    if (!Array.isArray(parsed)) return [];

    return parsed.filter((item): item is Conversation => {
      if (!isObject(item)
        || typeof item.id !== 'string'
        || typeof item.title !== 'string'
        || !Array.isArray(item.messages)
        || !item.messages.every(isStoredMessage)
        || !isTimestamp(item.createdAt)
        || !isTimestamp(item.updatedAt)) {
        return false;
      }
      return item.messages.some((message) => message.role === 'user' && message.text.trim().length > 0);
    }).sort((left, right) => right.updatedAt - left.updatedAt);
  } catch {
    return [];
  }
}

function conversationTitle(question: string): string {
  return question.trim() || 'New conversation';
}

function sortConversations(items: Conversation[]): Conversation[] {
  return [...items].sort((left, right) => right.updatedAt - left.updatedAt);
}

function buildConversation(
  id: string,
  messages: Message[],
  existing?: Conversation,
): Conversation | null {
  const firstQuestion = messages.find((message) => message.role === 'user' && message.text.trim());
  if (!firstQuestion) return null;
  return {
    id,
    title: conversationTitle(firstQuestion.text),
    messages,
    createdAt: existing?.createdAt ?? firstQuestion.timestamp,
    updatedAt: Date.now(),
  };
}

function upsertConversation(items: Conversation[], conversation: Conversation): Conversation[] {
  return sortConversations([
    ...items.filter((item) => item.id !== conversation.id),
    conversation,
  ]);
}

function App() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeConversationId, setActiveConversationId] = useState(
    () => conversations[0]?.id ?? createConversationId(),
  );
  const [messages, setMessages] = useState<Message[]>(
    () => conversations[0]?.messages ?? [createWelcomeMessage()],
  );
  const [historyStorageError, setHistoryStorageError] = useState('');
  const [categories, setCategories] = useState<string[]>([]);
  const [categoryCounts, setCategoryCounts] = useState<Record<string, number>>({});
  const [knowledge, setKnowledge] = useState<KnowledgeResponse>({
    items: [], total: 0, page: 1, page_size: PAGE_SIZE, has_more: false,
  });
  const [evaluation, setEvaluation] = useState<EvaluationResult | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [activeView, setActiveView] = useState<View>('chat');
  const [selectedCategory, setSelectedCategory] = useState('');
  const [knowledgeInput, setKnowledgeInput] = useState('');
  const [knowledgeQuery, setKnowledgeQuery] = useState('');
  const [knowledgePage, setKnowledgePage] = useState(1);
  const [draft, setDraft] = useState('');
  const [mode, setMode] = useState<'tfidf' | 'semantic' | 'hybrid'>('semantic');
  const [loading, setLoading] = useState(false);
  const [loadingCollection, setLoadingCollection] = useState(false);
  const [error, setError] = useState('');
  const [navOpen, setNavOpen] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const [feedbackSent, setFeedbackSent] = useState<Record<number, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [messages, loading]);

  useEffect(() => {
    const existing = conversations.find((conversation) => conversation.id === activeConversationId);
    const updated = buildConversation(activeConversationId, messages, existing);
    if (updated) setConversations((current) => upsertConversation(current, updated));
  }, [activeConversationId, messages]);

  useEffect(() => {
    try {
      window.localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(conversations));
      setHistoryStorageError('');
    } catch {
      setHistoryStorageError('Chat history could not be saved in this browser.');
    }
  }, [conversations]);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetchJson<{ status: string; corpus_documents: number; semantic_model_loaded: boolean; semantic_model_error: string | null; mode: string }>('/api/health'),
      fetchJson<string[]>('/api/categories'),
      fetchJson<Record<string, number>>('/api/category-counts'),
    ]).then(([, categoryList, counts]) => {
      if (!active) return;
      setCategories(categoryList);
      setCategoryCounts(counts);
    }).catch(() => {
      if (active) {
        setError('The backend is unavailable. Start the API to load the BEC knowledge base.');
      }
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (activeView !== 'knowledge') return;
    const params = new URLSearchParams({
      page: String(knowledgePage),
      page_size: String(PAGE_SIZE),
      mode,
    });
    if (selectedCategory) params.set('category', selectedCategory);
    if (knowledgeQuery) params.set('query', knowledgeQuery);
    let active = true;
    fetchJson<KnowledgeResponse>(`/api/knowledge?${params.toString()}`)
      .then((data) => { if (active) setKnowledge(data); })
      .catch((reason: Error) => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [activeView, knowledgePage, knowledgeQuery, mode, selectedCategory]);

  useEffect(() => {
    if (activeView !== 'evaluation') return;
    let active = true;
    setEvaluationLoading(true);
    fetchJson<EvaluationResult>('/api/evaluation')
      .then((result) => { if (active) setEvaluation(result); })
      .catch((reason: Error) => { if (active) setError(reason.message); })
      .finally(() => { if (active) setEvaluationLoading(false); });
    return () => { active = false; };
  }, [activeView]);

  async function askQuestion(question: string, page = 1, updateMessageId?: number) {
    const cleanQuestion = question.trim();
    if (!cleanQuestion || loading) return;
    setError('');
    setDraft('');
    setLoading(true);
    if (!updateMessageId) {
      setMessages((current) => [...current, {
        id: Date.now(),
        role: 'user',
        text: cleanQuestion,
        timestamp: Date.now(),
      }]);
    } else {
      setLoadingCollection(true);
    }

    try {
      const result = await sendChatRequest({
        question: cleanQuestion,
        mode,
        page,
        page_size: PAGE_SIZE,
        history: updateMessageId
          ? []
          : messages.filter((message) => message.role === 'user').slice(-1).map((message) => message.text),
      });
      if (updateMessageId) {
        setMessages((current) => current.map((message) => message.id === updateMessageId
          ? { ...message, text: result.answer, sources: result.sources, result }
          : message));
      } else {
        setMessages((current) => [...current, {
          id: Date.now() + 1,
          role: 'assistant',
          text: result.answer,
          sources: result.sources,
          question: cleanQuestion,
          result,
          timestamp: Date.now() + 1,
        }]);
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The assistant could not answer this request.');
    } finally {
      setLoading(false);
      setLoadingCollection(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void askQuestion(draft);
  }

  function handleInputKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void askQuestion(draft);
    }
  }

  async function sendFeedback(message: Message, helpful: boolean) {
    if (!message.question || feedbackSent[message.id]) return;
    try {
      await fetchJson('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: message.question,
          answer: message.text,
          helpful,
          category: message.sources?.[0]?.category,
        }),
      });
      setFeedbackSent((current) => ({ ...current, [message.id]: true }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Feedback could not be saved.');
    }
  }

  async function copyAnswer(message: Message) {
    try {
      await navigator.clipboard.writeText(message.text);
      setCopied(message.id);
      window.setTimeout(() => setCopied(null), 1800);
    } catch {
      setError('Could not copy the answer. Check your browser clipboard permissions.');
    }
  }

  function persistCurrentConversation() {
    const existing = conversations.find((conversation) => conversation.id === activeConversationId);
    const updated = buildConversation(activeConversationId, messages, existing);
    if (!updated) return;

    const next = upsertConversation(conversations, updated);
    setConversations(next);
    try {
      window.localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(next));
      setHistoryStorageError('');
    } catch {
      setHistoryStorageError('Chat history could not be saved in this browser.');
    }
  }

  function startNewChat() {
    if (loading) return;
    persistCurrentConversation();
    setActiveConversationId(createConversationId());
    setMessages([createWelcomeMessage()]);
    setError('');
    setFeedbackSent({});
  }

  function openConversation(conversation: Conversation) {
    if (loading) return;
    persistCurrentConversation();
    setActiveConversationId(conversation.id);
    setMessages(conversation.messages);
    setFeedbackSent({});
    setError('');
    setActiveView('chat');
    setNavOpen(false);
  }

  function deleteConversation(conversation: Conversation) {
    if (loading || !window.confirm(`Delete "${conversation.title}" from chat history?`)) return;
    const remaining = conversations.filter((item) => item.id !== conversation.id);
    setConversations(remaining);
    if (conversation.id === activeConversationId) {
      setActiveConversationId(createConversationId());
      setMessages([createWelcomeMessage()]);
      setFeedbackSent({});
      setError('');
    }
  }

  function openCategory(category: string) {
    setSelectedCategory(category);
    setKnowledgePage(1);
    setKnowledgeInput('');
    setKnowledgeQuery('');
    setActiveView('knowledge');
    setNavOpen(false);
  }

  function selectView(view: View) {
    setActiveView(view);
    setNavOpen(false);
  }

  function handleKnowledgeSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setKnowledgePage(1);
    setKnowledgeQuery(knowledgeInput.trim());
  }

  const activeTitle: Record<View, string> = {
    chat: 'Ask Campus AI',
    knowledge: 'Knowledge Base',
    categories: 'Academic Categories',
    evaluation: 'Retrieval Evaluation',
    about: 'About the Project',
  };

  return (
    <main className="app-shell">
      {navOpen && <button className="nav-scrim" aria-label="Close navigation" onClick={() => setNavOpen(false)} type="button" />}
      <aside className={`sidebar ${navOpen ? 'sidebar-open' : ''}`}>
        <a className="brand" href="#home" onClick={(event) => { event.preventDefault(); selectView('chat'); }} aria-label="CampusConnect home">
          <span className="brand-mark"><GraduationCap size={21} /></span>
          <span className="brand-name">Campus<span>Connect</span><small>AI</small><em>Your AI-powered campus assistant</em></span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <button className={`nav-item ${activeView === 'chat' ? 'nav-item-active' : ''}`} onClick={() => selectView('chat')} type="button">
          <MessageCircle size={17} /><span>Ask Campus AI</span>
        </button>
        <button className={`nav-item ${activeView === 'knowledge' ? 'nav-item-active' : ''}`} onClick={() => selectView('knowledge')} type="button">
          <BookOpen size={17} /><span>Knowledge Base</span>
        </button>
        <button className={`nav-item ${activeView === 'categories' ? 'nav-item-active' : ''}`} onClick={() => selectView('categories')} type="button">
          <CalendarDays size={17} /><span>Academic Categories</span>
        </button>
        <button className={`nav-item ${activeView === 'evaluation' ? 'nav-item-active' : ''}`} onClick={() => selectView('evaluation')} type="button">
          <Activity size={17} /><span>Retrieval Evaluation</span>
        </button>
        <button className={`nav-item ${activeView === 'about' ? 'nav-item-active' : ''}`} onClick={() => selectView('about')} type="button">
          <Sparkles size={17} /><span>About Project</span>
        </button>
        <section className="history-section" aria-label="Chat History">
          <div className="history-heading"><History size={16} /><span>Chat History</span></div>
          {conversations.length > 0 ? (
            <div className="history-list">
              {conversations.map((conversation) => (
                <div className={`history-item ${conversation.id === activeConversationId ? 'history-item-active' : ''}`} key={conversation.id}>
                  <button
                    className="history-open"
                    aria-current={conversation.id === activeConversationId ? 'page' : undefined}
                    disabled={loading}
                    onClick={() => openConversation(conversation)}
                    title={conversation.title}
                    type="button"
                  >
                    <span className="history-title">{conversation.title}</span>
                    <time dateTime={new Date(conversation.updatedAt).toISOString()}>
                      {new Date(conversation.updatedAt).toLocaleDateString([], { month: 'short', day: 'numeric' })}
                    </time>
                  </button>
                  <button
                    aria-label={`Delete conversation: ${conversation.title}`}
                    className="history-delete"
                    disabled={loading}
                    onClick={() => deleteConversation(conversation)}
                    title="Delete conversation"
                    type="button"
                  ><Trash2 size={14} /></button>
                </div>
              ))}
            </div>
          ) : (
            <p className="history-empty">Your conversations will appear here.</p>
          )}
          {historyStorageError && <p className="history-storage-error" role="status">{historyStorageError}</p>}
        </section>
        <div className="sidebar-bottom">
          <div className="sidebar-caption">Student support, made easier.</div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="topbar-leading">
            <button className="menu-button" onClick={() => setNavOpen((open) => !open)} type="button" aria-label={navOpen ? 'Close menu' : 'Open menu'}>
              {navOpen ? <X size={19} /> : <Menu size={19} />}
            </button>
            <div className="breadcrumb"><span>CampusConnect</span><b>/</b>{activeTitle[activeView]}</div>
          </div>
          <div className="topbar-actions">
            <span className="retrieval-mode-label">{mode === 'tfidf' ? 'TF-IDF' : mode === 'semantic' ? 'Semantic' : 'Hybrid'} mode</span>
            <span className="demo-label">BEC KNOWLEDGE BASE</span>
            {activeView === 'chat' && <button className="clear-button" disabled={loading} onClick={startNewChat} title="Start a new chat" type="button"><Plus size={16} /><span>New Chat</span></button>}
          </div>
        </header>

        <div className="conversation">
          {activeView === 'chat' && <>
            <div className="intro-block">
              <div className="eyebrow"><Sparkles size={14} /> BEC STUDENT SUPPORT</div>
              <h1>How can I help<br /><span>you today?</span></h1>
              <p>Explore information about Basaveshwar Engineering College, Bagalkote. Dataset entries were supplied by the project owner and have not been independently verified; confirm important or current details with the college.</p>
            </div>
            <div className="messages" aria-live="polite">
              {messages.map((message) => (
                <article className={`message message-${message.role} ${message.result?.status === 'unsupported' ? 'message-fallback' : ''}`} key={message.id}>
                  {message.role === 'assistant' && <span className="assistant-avatar"><Bot size={17} /></span>}
                  <div className="message-content">
                    <div className="message-heading">
                      <div className="message-author">{message.role === 'assistant' ? 'CampusConnect AI' : 'You'}</div>
                      <time dateTime={new Date(message.timestamp).toISOString()}><Clock3 size={12} />{new Date(message.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</time>
                    </div>
                    {message.result?.status === 'unsupported' && <span className="fallback-label">Clarification needed</span>}
                    <p className="message-text">{message.text}</p>
                    {message.result?.intent === 'collection' && message.result.records.length > 0 && <>
                      <div className="collection-header">{message.result.total} FAQ records · page {message.result.page}</div>
                      <details className="collection-browse">
                        <summary>Browse matching FAQ records</summary>
                        <div className="chat-record-list">
                          {message.result.records.map((record) => (
                            <article className="chat-record" key={record.document_id}>
                              <div className="record-meta"><span>{record.category}</span><code>{record.document_id}</code></div>
                              <h3>{record.title}</h3>
                              <p>{record.answer}</p>
                              <small>{record.source_file}</small>
                            </article>
                          ))}
                        </div>
                      </details>
                      <div className="pagination-controls">
                        <button
                          disabled={message.result.page <= 1 || loadingCollection}
                          onClick={() => void askQuestion(message.question || '', message.result!.page - 1, message.id)}
                          type="button"
                        ><ArrowUp size={15} /> Previous</button>
                        <span>{message.result.page} / {Math.max(1, Math.ceil(message.result.total / message.result.page_size))}</span>
                        <button
                          disabled={!message.result.has_more || loadingCollection}
                          onClick={() => void askQuestion(message.question || '', message.result!.page + 1, message.id)}
                          type="button"
                        >Next <ArrowDown size={15} /></button>
                      </div>
                    </>}
                    {message.role === 'assistant'
                      && message.result?.status === 'answered'
                      && message.result.follow_up_questions.length > 0 && (
                        <details className="related-faq-panel">
                          <summary>Related FAQs ({message.result.follow_up_questions.length})</summary>
                          <div className="related-faq-list">
                            {message.result.follow_up_questions.map((relatedQuestion) => (
                              <button
                                key={relatedQuestion}
                                disabled={loading}
                                onClick={() => void askQuestion(relatedQuestion)}
                                type="button"
                              >
                                {relatedQuestion}
                              </button>
                            ))}
                          </div>
                        </details>
                      )}
                    {message.sources && message.sources.length > 0 && (
                      <details className="source-panel">
                        <summary>Sources ({message.sources.length})</summary>
                        <div className="source-list">
                          {message.sources.map((source, index) => (
                            <div className="source-item" key={`${source.document_id || source.title}-${index}`}>
                              <BookOpen size={13} />
                              <div className="source-details">
                                {source.matched_question && <strong>{source.matched_question}</strong>}
                                {source.category && <span className="source-category">{source.category}</span>}
                                {source.url && <a href={source.url} target="_blank" rel="noreferrer">View source <ExternalLink size={12} /></a>}
                                {source.verification_note && <small>{source.verification_note}</small>}
                                {source.verification_status && <small>{source.verification_status.replace(/_/g, ' ')}</small>}
                                {source.source_page_label && <small>{source.source_page_label}</small>}
                              </div>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                    {message.role === 'assistant' && message.question && (
                      <div className="feedback-actions" aria-label="Answer actions">
                        <button onClick={() => void copyAnswer(message)} title="Copy answer" type="button">
                          {copied === message.id ? <Check size={14} /> : <Clipboard size={14} />}
                        </button>
                        {feedbackSent[message.id] ? <span>Thanks for your feedback</span> : <>
                          <span>Helpful?</span>
                          <button onClick={() => void sendFeedback(message, true)} aria-label="Helpful answer" type="button">Yes</button>
                          <button onClick={() => void sendFeedback(message, false)} aria-label="Not helpful answer" type="button">No</button>
                        </>}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {loading && <div className="loading-line"><span className="spinner" /> Searching BEC records…</div>}
              <div ref={messagesEndRef} />
            </div>
            {messages.length === 1 && (
              <section className="suggestion-section">
                <div className="suggestions-heading">TRY ASKING</div>
                <div className="suggestion-grid">
                  {SUGGESTED_QUESTIONS.map((question) => (
                    <button className="suggestion-card" key={question} onClick={() => void askQuestion(question)} type="button">
                      <span className="suggestion-icon"><MessageCircle size={16} /></span>
                      <span><b>{question}</b><small>Ask the BEC knowledge base</small></span>
                      <ArrowDown size={14} />
                    </button>
                  ))}
                </div>
              </section>
            )}
            {error && <div className="error-message" role="alert">{error}</div>}
            <div className="composer-wrap">
              <div className="mode-switch" aria-label="Retrieval mode">
                <button className={mode === 'tfidf' ? 'mode-active' : ''} onClick={() => setMode('tfidf')} type="button">Keyword</button>
                <button className={mode === 'semantic' ? 'mode-active' : ''} onClick={() => setMode('semantic')} type="button">Semantic</button>
                <button className={mode === 'hybrid' ? 'mode-active' : ''} onClick={() => setMode('hybrid')} type="button">Hybrid</button>
              </div>
              <form className="composer" onSubmit={handleSubmit}>
                <textarea
                  aria-label="Ask a question"
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={handleInputKeyDown}
                  placeholder="Ask about courses, faculty, schedules, or policies… (Enter to send)"
                  rows={1}
                  value={draft}
                />
                <button aria-label="Send question" className="send-button" disabled={!draft.trim() || loading} type="submit"><Send size={17} /></button>
              </form>
              <div className="disclaimer">Answers use user-provided records that have not been independently verified. Confirm important information with BEC.</div>
            </div>
          </>}

          {activeView === 'knowledge' && <section className="page-view">
            <div className="eyebrow"><BookOpen size={14} /> ACADEMIC CORPUS</div>
            <h1>Knowledge <span>Base</span></h1>
            <p className="page-intro">Search and browse the BEC question-and-answer records used by the assistant. These user-provided records have not been independently verified and may not reflect current college policy.</p>
            <form className="knowledge-search" onSubmit={handleKnowledgeSearch}>
              <Search size={17} />
              <input aria-label="Search knowledge base" onChange={(event) => setKnowledgeInput(event.target.value)} placeholder="Search academic records…" value={knowledgeInput} />
              <button type="submit">Search</button>
            </form>
            <div className="knowledge-filters">
              <button className={!selectedCategory ? 'filter-active' : ''} onClick={() => { setSelectedCategory(''); setKnowledgePage(1); }} type="button">All categories</button>
              {categories.map((category) => <button className={selectedCategory === category ? 'filter-active' : ''} key={category} onClick={() => { setSelectedCategory(category); setKnowledgePage(1); }} type="button">{category} <span>{categoryCounts[category] ?? 0}</span></button>)}
            </div>
            <div className="knowledge-count">{knowledge.total} RECORDS · PAGE {knowledge.page}</div>
            <div className="knowledge-list">
              {knowledge.items.map((record) => (
                <article className="knowledge-record" key={record.document_id}>
                  <div className="record-meta"><span>{record.category}</span><code>{record.document_id}</code></div>
                  <h2>{record.title}</h2>
                  <dl className="record-fields">
                    {Object.entries(record.original_record).filter(([key]) => key !== 'document_id').map(([key, value]) => (
                      <div key={key}><dt>{key.replace(/_/g, ' ')}</dt><dd>{displayValue(value)}</dd></div>
                    ))}
                  </dl>
                  <div className="record-source">{record.source_file}</div>
                </article>
              ))}
              {!knowledge.items.length && <div className="knowledge-empty">No records found. Clear filters or try another search.</div>}
            </div>
            <div className="pagination-controls page-pagination">
              <button disabled={knowledgePage <= 1} onClick={() => setKnowledgePage((page) => Math.max(1, page - 1))} type="button"><ArrowUp size={15} /> Previous</button>
              <span>Page {knowledge.page} of {Math.max(1, Math.ceil(knowledge.total / PAGE_SIZE))}</span>
              <button disabled={!knowledge.has_more} onClick={() => setKnowledgePage((page) => page + 1)} type="button">Next <ArrowDown size={15} /></button>
            </div>
          </section>}

          {activeView === 'categories' && <section className="page-view">
            <div className="eyebrow"><CalendarDays size={14} /> BROWSE THE DATASET</div>
            <h1>Academic <span>Categories</span></h1>
            <p className="page-intro">Choose a collection to browse its records. Counts come from the loaded dataset.</p>
            <div className="category-grid">
              {categories.map((category) => (
                <button className="category-card" key={category} onClick={() => openCategory(category)} type="button">
                  <span className="suggestion-icon"><BookOpen size={18} /></span>
                  <b>{category}</b>
                  <small>{CATEGORY_DESCRIPTIONS[category] || 'Browse this academic collection.'}</small>
                  <span className="category-count">{categoryCounts[category] ?? 0} records</span>
                </button>
              ))}
            </div>
            {error && <div className="error-message" role="alert">{error}</div>}
          </section>}

          {activeView === 'evaluation' && <section className="page-view">
            <div className="eyebrow"><Activity size={14} /> MEASURED RETRIEVAL QUALITY</div>
            <h1>Retrieval <span>Evaluation</span></h1>
            <p className="page-intro">These measurements use a small, manually prepared BEC FAQ benchmark. They are useful for regression checks, not a guarantee of real-world accuracy.</p>
            {evaluationLoading && <div className="evaluation-state">Loading evaluation results…</div>}
            {!evaluationLoading && evaluation?.status === 'not_run' && (
              <div className="evaluation-state">
                <h2>No matching evaluation report is available</h2>
                <p>{evaluation.message}</p>
                <code>cd backend; ..\.venv\Scripts\python.exe -m scripts.evaluate_retrieval</code>
              </div>
            )}
            {!evaluationLoading && evaluation?.metrics && <>
              <div className="evaluation-summary">
                <span><b>{evaluation.answerable_query_count ?? '—'}</b> answerable queries</span>
                <span><b>{evaluation.rejection_query_count ?? '—'}</b> rejection queries</span>
                <span><b>{evaluation.knowledge_documents ?? '—'}</b> searchable documents</span>
                <span><b>{evaluation.semantic_model_available ? 'Loaded' : 'Unavailable'}</b> semantic model</span>
              </div>
              <div className="evaluation-table-wrap">
                <table className="evaluation-table">
                  <thead><tr><th>Method</th><th>Top-1 accuracy</th><th>Top-3 accuracy</th><th>MRR</th><th>Rejection accuracy</th></tr></thead>
                  <tbody>
                    {Object.entries(evaluation.metrics).map(([method, metric]) => (
                      <tr key={method}>
                        <th scope="row">{method.toUpperCase()}</th>
                        {typeof metric.top1_accuracy === 'number' ? <>
                          <td>{((metric.top1_accuracy ?? 0) * 100).toFixed(1)}%</td>
                          <td>{((metric.top3_accuracy ?? 0) * 100).toFixed(1)}%</td>
                          <td>{((metric.mrr ?? 0) * 100).toFixed(1)}%</td>
                          <td>{metric.rejection_accuracy === undefined ? '—' : `${(metric.rejection_accuracy * 100).toFixed(1)}%`}</td>
                        </> : <td colSpan={4}>{metric.status}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {evaluation.limitations?.map((note) => <p className="evaluation-note" key={note}>{note}</p>)}
            </>}
            {error && <div className="error-message" role="alert">{error}</div>}
          </section>}

          {activeView === 'about' && <section className="page-view about-view">
            <div className="eyebrow"><Sparkles size={14} /> PROJECT OVERVIEW</div>
            <h1>About <span>CampusConnect AI</span></h1>
            <p className="page-intro">A mini-project demonstrating how academic information can be searched and presented with traceable sources.</p>
            <div className="about-grid">
              <article><h2>Problem</h2><p>Students may need to search multiple information sources to find policies, course details, schedules, and campus services. This demo provides one natural-language search interface.</p></article>
              <article><h2>Technology</h2><p>React, TypeScript, Vite, Python, FastAPI, scikit-learn TF-IDF, Sentence Transformers, and SQLite for feedback and analytics.</p></article>
              <article><h2>Retrieval</h2><p>Keyword retrieval matches terms. Semantic search compares pretrained text embeddings. Hybrid search combines ranked results using Reciprocal Rank Fusion.</p></article>
              <article><h2>Responsible use</h2><p>This knowledge base concerns Basaveshwar Engineering College, Bagalkote. Its records were user-provided and have not been independently verified; confirm important or current details with the appropriate college office.</p></article>
              <article><h2>Evaluation</h2><p>A manually prepared BEC benchmark reports retrieval and rejection metrics. It is small and should not be treated as a guarantee of real-world accuracy.</p></article>
              <article><h2>RAG status</h2><p>An optional LLM can format retrieved evidence if configured. Without it, the app uses a deterministic local response builder; it does not require a paid LLM service.</p></article>
            </div>
            <div className="demo-notice">BEC dataset notice: information was supplied for this project and has not been independently verified. The dataset contains internal metadata that conflicts with some of its records; treat responses as informational only and confirm them with BEC.</div>
          </section>}
        </div>
      </section>
    </main>
  );
}

export default App;
