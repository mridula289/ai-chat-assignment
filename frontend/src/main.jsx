import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

function App() {
  const [sessionId] = useState(() => `s-${crypto.randomUUID()}`);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  async function sendMessage(event) {
    event?.preventDefault();
    const message = input.trim();
    if (!message || sending) return;
    setSending(true);
    setMessages((current) => [...current, { role: "user", content: message }]);
    setInput("");
    try {
      const body = { session_id: sessionId, message };
      if (systemPrompt.trim()) body.system_prompt = systemPrompt.trim();
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The request failed.");
      setMessages((current) => [...current, {
        role: "assistant",
        content: data.reply,
        metadata: data.metadata,
      }]);
    } catch (error) {
      setMessages((current) => [...current, {
        role: "error",
        content: `Unable to send message: ${error.message}`,
      }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <main className="app">
      <header>
        <div className="brand">Lumen <span>AI</span></div>
        <div className="session">Session <code>{sessionId}</code></div>
      </header>
      <section className="workspace">
        <aside>
          <label htmlFor="system">System prompt</label>
          <textarea
            id="system"
            data-testid="system-prompt-input"
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            placeholder="Optional session instructions..."
          />
        </aside>
        <div className="chat">
          <div className="messages">
            {messages.length === 0 && <div className="empty">Start a conversation</div>}
            {messages.map((message, index) => (
              message.role === "assistant" ? (
                <article className="assistant" data-testid="assistant-message" key={index}>
                  <div className="avatar">AI</div>
                  <div className="bubble">
                    <p>{message.content}</p>
                    <div className="metadata">
                      <span data-testid="meta-source">{message.metadata.resolution_source}</span>
                      <span data-testid="meta-model">{message.metadata.model}</span>
                      <span data-testid="meta-latency">{message.metadata.latency_ms} ms</span>
                      <span>Input: <b data-testid="meta-input-tokens">{message.metadata.input_tokens}</b></span>
                      <span>Output: <b data-testid="meta-output-tokens">{message.metadata.output_tokens}</b></span>
                    </div>
                  </div>
                </article>
              ) : (
                <article className={message.role === "error" ? "error" : "user"} key={index}>
                  {message.role === "user" ? <div className="user-bubble">{message.content}</div> : message.content}
                </article>
              )
            ))}
            <div ref={endRef} />
          </div>
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              data-testid="chat-input"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) sendMessage(event);
              }}
              placeholder="Type a message..."
              rows="2"
            />
            <button data-testid="send-button" type="submit" disabled={sending || !input.trim()}>
              {sending ? "Sending..." : "Send"}
            </button>
          </form>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
