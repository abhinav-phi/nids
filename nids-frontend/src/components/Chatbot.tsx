import { useState, useRef, useEffect, useCallback } from "react";
import { sendChatMessage } from "@/api/client";
import { useIsMobile } from "@/hooks/use-mobile";

/* ── Types ────────────────────────────────────────────────── */

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

/* ── Markdown-lite renderer ───────────────────────────────── */

function renderMarkdown(text: string) {
  // Convert **bold**
  let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  // Convert `code`
  html = html.replace(/`([^`]+)`/g, '<code class="chat-inline-code">$1</code>');
  // Convert bullet points
  html = html.replace(/^[•●]\s?/gm, "• ");
  // Convert newlines to <br/>
  html = html.replace(/\n/g, "<br/>");
  return html;
}

/* ── Typing indicator ─────────────────────────────────────── */

const TypingDots = () => (
  <div className="flex items-center gap-1 px-4 py-3">
    <div className="flex gap-1">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-2 h-2 rounded-full"
          style={{
            backgroundColor: "#a1faff",
            opacity: 0.6,
            animation: `chatDotBounce 1.4s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
    <span className="ml-2 text-xs" style={{ color: "rgba(255,255,255,0.35)" }}>
      Sentinel AI is thinking…
    </span>
  </div>
);

/* ── Suggested questions ──────────────────────────────────── */

const SUGGESTIONS = [
  "Show top 5 attacker IPs in the last 24 hours",
  "Summarize critical alerts from today",
  "Explain port scanning in network traffic",
  "What do NIDS severity levels mean?",
];

/* ── Main Component ───────────────────────────────────────── */

const Chatbot = () => {
  const isMobile = useIsMobile();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  /* Auto-scroll to bottom */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  /* Focus input when opened */
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 200);
    }
  }, [isOpen]);

  /* Build history for API */
  const buildHistory = useCallback(() => {
    return messages.map((m) => ({ role: m.role, content: m.content }));
  }, [messages]);

  /* Send message */
  const handleSend = useCallback(
    async (text?: string) => {
      const msg = (text ?? input).trim();
      if (!msg || isLoading) return;

      const userMsg: Message = {
        id: `u-${Date.now()}`,
        role: "user",
        content: msg,
      };

      setMessages((prev) => [...prev, userMsg]);
      setInput("");
      setIsLoading(true);

      try {
        const history = buildHistory();
        const { reply } = await sendChatMessage(msg, history);

        const aiMsg: Message = {
          id: `a-${Date.now()}`,
          role: "assistant",
          content: reply,
        };
        setMessages((prev) => [...prev, aiMsg]);
      } catch (err: any) {
        const errorText =
          err?.response?.data?.detail ??
          "Sorry, I couldn't reach the AI service. Please try again.";
        const errMsg: Message = {
          id: `e-${Date.now()}`,
          role: "assistant",
          content: `⚠️ ${errorText}`,
        };
        setMessages((prev) => [...prev, errMsg]);
      } finally {
        setIsLoading(false);
      }
    },
    [input, isLoading, buildHistory]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  /* ── Render ───────────────────────────────────────────── */

  const fabSize = isMobile ? 54 : 60;
  const fabBottom = isMobile ? 16 : 28;
  const fabRight = isMobile ? 14 : 28;
  const windowBottom = isMobile ? 82 : 100;
  const windowRight = isMobile ? 10 : 28;
  const windowWidth = isMobile ? "calc(100vw - 20px)" : 400;
  const windowHeight = isMobile ? "min(74vh, 560px)" : 560;

  return (
    <>
      {/* Keyframe injection */}
      <style>{`
        @keyframes chatDotBounce {
          0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
          40% { transform: scale(1); opacity: 1; }
        }
        @keyframes chatSlideUp {
          from { opacity: 0; transform: translateY(20px) scale(0.95); }
          to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes chatPulseRing {
          0%   { box-shadow: 0 0 0 0 rgba(161, 250, 255, 0.4); }
          70%  { box-shadow: 0 0 0 12px rgba(161, 250, 255, 0); }
          100% { box-shadow: 0 0 0 0 rgba(161, 250, 255, 0); }
        }
        .chat-inline-code {
          background: rgba(161, 250, 255, 0.1);
          color: #a1faff;
          padding: 1px 5px;
          border-radius: 4px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 0.85em;
        }
      `}</style>

      {/* ── FAB Button ──────────────────────────────────── */}
      <button
        id="chatbot-fab"
        onClick={() => setIsOpen((o) => !o)}
        className="fixed z-50 flex items-center justify-center transition-all duration-300 group"
        style={{
          bottom: fabBottom,
          right: fabRight,
          width: fabSize,
          height: fabSize,
          borderRadius: "50%",
          background: isOpen
            ? "linear-gradient(135deg, #ff716c 0%, #e04040 100%)"
            : "linear-gradient(135deg, #a1faff 0%, #699cff 50%, #ac8aff 100%)",
          boxShadow: isOpen
            ? "0 4px 24px rgba(255, 113, 108, 0.4)"
            : "0 4px 24px rgba(161, 250, 255, 0.3)",
          animation: !isOpen ? "chatPulseRing 2.5s infinite" : "none",
          border: "none",
          cursor: "pointer",
        }}
      >
        {isOpen ? (
          /* Close icon */
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          /* Chat icon */
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            <line x1="9" y1="10" x2="9" y2="10" strokeWidth="3" strokeLinecap="round" />
            <line x1="12" y1="10" x2="12" y2="10" strokeWidth="3" strokeLinecap="round" />
            <line x1="15" y1="10" x2="15" y2="10" strokeWidth="3" strokeLinecap="round" />
          </svg>
        )}
      </button>

      {/* ── Chat Window ─────────────────────────────────── */}
      {isOpen && (
        <div
          id="chatbot-window"
          className="fixed z-50 flex flex-col"
          style={{
            bottom: windowBottom,
            right: windowRight,
            width: windowWidth,
            maxWidth: 400,
            height: windowHeight,
            borderRadius: 16,
            background: "linear-gradient(180deg, #111827 0%, #0d1117 100%)",
            border: "1px solid rgba(161, 250, 255, 0.12)",
            boxShadow:
              "0 24px 80px rgba(0,0,0,0.6), 0 0 1px rgba(161,250,255,0.2), inset 0 1px 0 rgba(255,255,255,0.05)",
            animation: "chatSlideUp 0.3s ease-out",
            overflow: "hidden",
          }}
        >
          {/* ── Header ────────────────────────────────── */}
          <div
            className="flex items-center gap-3 px-5 py-4 shrink-0"
            style={{
              background: "linear-gradient(135deg, rgba(161,250,255,0.08) 0%, rgba(172,138,255,0.06) 100%)",
              borderBottom: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            <div
              className="w-9 h-9 rounded-lg flex items-center justify-center"
              style={{
                background: "linear-gradient(135deg, #a1faff 0%, #699cff 100%)",
                boxShadow: "0 2px 12px rgba(161, 250, 255, 0.3)",
              }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#0a0e19" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a4 4 0 0 1 4 4v2a4 4 0 0 1-8 0V6a4 4 0 0 1 4-4z" />
                <path d="M5 10v2a7 7 0 0 0 14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="22" />
                <line x1="8" y1="22" x2="16" y2="22" />
              </svg>
            </div>
            <div className="flex-1">
              <h3
                className="text-sm font-semibold tracking-wide"
                style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
              >
                Sentinel AI
              </h3>
              <p className="text-[10px] font-medium tracking-widest uppercase" style={{ color: "rgba(161,250,255,0.6)" }}>
                Network Security Assistant
              </p>
            </div>
            <button
              onClick={() => {
                setMessages([]);
                setInput("");
              }}
              className="p-1.5 rounded-md transition-colors"
              style={{ color: "rgba(255,255,255,0.3)" }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "#ff716c")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.3)")}
              title="Clear conversation"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="3 6 5 6 21 6" />
                <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                <path d="M10 11v6" />
                <path d="M14 11v6" />
              </svg>
            </button>
          </div>

          {/* ── Messages Area ─────────────────────────── */}
          <div
            className="flex-1 overflow-y-auto px-4 py-4 space-y-3"
            style={{
              scrollbarWidth: "thin",
              scrollbarColor: "rgba(255,255,255,0.1) transparent",
            }}
          >
            {/* Welcome message when empty */}
            {messages.length === 0 && !isLoading && (
              <div className="flex flex-col items-center justify-center h-full gap-4 py-6">
                <div
                  className="w-14 h-14 rounded-2xl flex items-center justify-center"
                  style={{
                    background: "linear-gradient(135deg, rgba(161,250,255,0.12) 0%, rgba(172,138,255,0.08) 100%)",
                    border: "1px solid rgba(161,250,255,0.1)",
                  }}
                >
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#a1faff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <path d="M12 16v-4" />
                    <path d="M12 8h.01" />
                  </svg>
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium" style={{ color: "#e8eafb" }}>
                    Hello! I'm Sentinel AI
                  </p>
                  <p className="text-xs mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
                    Ask about network attacks or your NIDS data,
                    <br />
                    and I will ground answers in dashboard alerts.
                  </p>
                </div>
                {/* Suggested prompts */}
                <div className="w-full space-y-2 mt-2 px-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => handleSend(s)}
                      className="w-full text-left px-3 py-2.5 rounded-lg text-xs transition-all duration-200"
                      style={{
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.06)",
                        color: "rgba(255,255,255,0.55)",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background = "rgba(161,250,255,0.06)";
                        e.currentTarget.style.borderColor = "rgba(161,250,255,0.2)";
                        e.currentTarget.style.color = "#a1faff";
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                        e.currentTarget.style.borderColor = "rgba(255,255,255,0.06)";
                        e.currentTarget.style.color = "rgba(255,255,255,0.55)";
                      }}
                    >
                      → {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages.map((m) => (
              <div
                key={m.id}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className="max-w-[85%] rounded-xl px-4 py-3 text-[13px] leading-relaxed"
                  style={
                    m.role === "user"
                      ? {
                          background: "linear-gradient(135deg, #699cff 0%, #ac8aff 100%)",
                          color: "#fff",
                          borderBottomRightRadius: 4,
                          boxShadow: "0 2px 12px rgba(105, 156, 255, 0.3)",
                        }
                      : {
                          background: "rgba(255,255,255,0.04)",
                          border: "1px solid rgba(255,255,255,0.06)",
                          color: "#d1d5e8",
                          borderBottomLeftRadius: 4,
                        }
                  }
                >
                  {m.role === "assistant" ? (
                    <div dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }} />
                  ) : (
                    m.content
                  )}
                </div>
              </div>
            ))}

            {isLoading && <TypingDots />}
            <div ref={messagesEndRef} />
          </div>

          {/* ── Input Area ────────────────────────────── */}
          <div
            className="shrink-0 px-4 py-3"
            style={{
              background: "rgba(17,24,39,0.95)",
              borderTop: "1px solid rgba(255,255,255,0.06)",
            }}
          >
            <div
              className="flex items-center gap-2 rounded-xl px-4 py-2"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            >
              <input
                ref={inputRef}
                id="chatbot-input"
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about network security…"
                disabled={isLoading}
                className="flex-1 bg-transparent text-sm outline-none placeholder:text-white/25"
                style={{ color: "#e8eafb", fontFamily: "'Inter', sans-serif" }}
              />
              <button
                id="chatbot-send"
                onClick={() => handleSend()}
                disabled={!input.trim() || isLoading}
                className="p-2 rounded-lg transition-all duration-200 flex items-center justify-center"
                style={{
                  background:
                    input.trim() && !isLoading
                      ? "linear-gradient(135deg, #a1faff 0%, #699cff 100%)"
                      : "rgba(255,255,255,0.05)",
                  opacity: input.trim() && !isLoading ? 1 : 0.4,
                  cursor: input.trim() && !isLoading ? "pointer" : "not-allowed",
                }}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke={input.trim() && !isLoading ? "#0a0e19" : "#fff"}
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <line x1="22" y1="2" x2="11" y2="13" />
                  <polygon points="22 2 15 22 11 13 2 9 22 2" />
                </svg>
              </button>
            </div>
            <p
              className="text-center mt-2 text-[10px]"
              style={{ color: "rgba(255,255,255,0.2)" }}
            >
              Powered by Gemini AI · Sentinel v1.0
            </p>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;
