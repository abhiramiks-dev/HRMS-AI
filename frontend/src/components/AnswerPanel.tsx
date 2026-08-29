interface AnswerPanelProps {
  question: string
  answer: string
  loading: boolean
  error: string | null
}

export function AnswerPanel({ question, answer, loading, error }: AnswerPanelProps) {
  return (
    <section className="card answer-card" aria-live="polite">
      <div className="section-kicker">ASSISTANT RESPONSE</div>
      {loading ? (
        <div className="loading-state"><span className="spinner" /> Searching HR knowledge…</div>
      ) : error ? (
        <div className="error-state"><strong>Unable to get an answer</strong><p>{error}</p></div>
      ) : answer ? (
        <>
          <div className="response-block">
            <span className="response-label">Question</span>
            <p className="question-text">{question}</p>
          </div>
          <div className="response-block answer-block">
            <span className="response-label">Answer</span>
            <p>{answer}</p>
          </div>
        </>
      ) : (
        <div className="empty-state">
          <div className="empty-icon">✦</div>
          <p>Your grounded HR answer will appear here.</p>
        </div>
      )}
    </section>
  )
}
