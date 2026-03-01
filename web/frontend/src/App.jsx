import { useState } from 'react'
import './App.css'

function App() {
  const [context, setContext] = useState("")
  const [question, setQuestion] = useState("")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const SAMPLES = [
    {
      title: "🏈 Super Bowl 50",
      context: "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.",
      question: "Which NFL team represented the AFC at Super Bowl 50?"
    },
    {
      title: "🌍 Amazon Rainforest",
      context: "The Amazon rainforest, alternatively the Amazon jungle, also known in English as Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 km2, of which 5,500,000 km2 are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 formally acknowledged indigenous territories.",
      question: "How many square kilometers of the Amazon basin are covered by the rainforest?"
    }
  ]

  const loadSample = (sample) => {
    setContext(sample.context)
    setQuestion(sample.question)
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!context.trim() || !question.trim()) {
      setError("Please provide both a context and a question.")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ context, question })
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || "Failed to fetch answer. Is the FastAPI backend running?")
    } finally {
      setLoading(false)
    }
  }

  // Highlight function
  const renderHighlightedContext = () => {
    if (!result || !result.answer) return <p className="text-secondary">{context || "No context provided yet."}</p>

    const { start_char, end_char } = result
    const before = context.slice(0, start_char)
    const highlight = context.slice(start_char, end_char)
    const after = context.slice(end_char)

    return (
      <p className="context-content">
        {before}
        <mark className="highlight-mark">{highlight}</mark>
        {after}
      </p>
    )
  }

  return (
    <div className="app-container">
      {/* Sidebar / Header */}
      <header className="hero-header">
        <div className="hero-content">
          <h1>🤖 QA Transformers</h1>
          <p>Powered by DistilBERT & React</p>
        </div>
      </header>

      <main className="main-layout">
        <aside className="sidebar">
          <h3>✨ Try a Sample</h3>
          <div className="sample-list">
            {SAMPLES.map((s, i) => (
              <button key={i} className="sample-btn" onClick={() => loadSample(s)}>
                {s.title}
              </button>
            ))}
          </div>

          <div className="info-box">
            <h4>ℹ️ How it works</h4>
            <p>This web app sends your text to a local FastAPI backend running a fine-tuned DistilBERT QA model.</p>
          </div>
        </aside>

        <section className="qa-workspace">
          <div className="input-group">
            <label>📄 Context Paragraph</label>
            <textarea
              value={context}
              onChange={e => setContext(e.target.value)}
              placeholder="Paste the context here..."
              rows={8}
            />
          </div>

          <div className="input-group">
            <label>❓ Your Question</label>
            <input
              type="text"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="What do you want to know?"
            />
          </div>

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? <span className="spinner">🧠 Thinking...</span> : "🚀 Find Answer"}
          </button>

          {error && (
            <div className="error-banner">
              ⚠️ {error}
            </div>
          )}

          {result && (
            <div className="result-container fade-in">
              <div className="answer-card">
                <h3>🎯 Answer Found</h3>
                <div className="answer-text">{result.answer}</div>

                <div className="confidence-meter">
                  <span>Confidence</span>
                  <div className="progress-bar-bg">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(0, result.confidence * 5))}%` }}
                    ></div>
                  </div>
                  <span className="conf-value">{(result.confidence).toFixed(2)}</span>
                </div>
              </div>

              <div className="context-highlight-card">
                <h3>📖 Context Highlight</h3>
                {renderHighlightedContext()}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
