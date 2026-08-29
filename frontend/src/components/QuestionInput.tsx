interface QuestionInputProps {
  question: string
  loading: boolean
  onChange: (question: string) => void
  onSubmit: () => void
}

const examples = [
  'How many annual leave days are employees entitled to?',
  'How much leave does E001 have?',
  'What is the emergency leave entitlement?',
]

export function QuestionInput({ question, loading, onChange, onSubmit }: QuestionInputProps) {
  return (
    <section className="card question-card">
      <div className="section-kicker">ASK THE ASSISTANT</div>
      <h2>What would you like to know?</h2>
      <textarea
        value={question}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) onSubmit()
        }}
        placeholder="Ask an HR question..."
        aria-label="Ask an HR question"
        rows={4}
      />
      <div className="input-footer">
        <span className="hint">Press Ctrl + Enter to submit</span>
        <button className="primary-button" onClick={onSubmit} disabled={loading || !question.trim()}>
          {loading ? 'Thinking…' : 'Ask Assistant'}
          {!loading && <span aria-hidden="true">→</span>}
        </button>
      </div>
      <div className="examples">
        <span>Try an example</span>
        {examples.map((example) => (
          <button key={example} className="example-button" onClick={() => onChange(example)}>
            {example}
          </button>
        ))}
      </div>
    </section>
  )
}
