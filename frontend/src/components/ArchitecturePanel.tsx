export function ArchitecturePanel() {
  return (
    <div className="info-grid">
      <section className="card info-card">
        <div className="section-kicker">AGENTIC ROUTING</div>
        <h2>One question, the right tool</h2>
        <p className="muted">Architecture explanation — the API response does not claim which tool was selected.</p>
        <div className="route"><span className="route-icon purple">⌕</span><div><strong>Policy questions</strong><span>PolicySearchTool → RAG → Gemini</span></div></div>
        <div className="route"><span className="route-icon teal">▣</span><div><strong>Employee questions</strong><span>EmployeeLeaveTool → Demo HR data</span></div></div>
      </section>
      <section className="card info-card">
        <div className="section-kicker">HOW IT WORKS</div>
        <h2>Grounded answers</h2>
        <div className="pipeline">
          {['Document', 'Chunking', 'Embeddings', 'ChromaDB', 'Semantic Retrieval', 'Gemini', 'Grounded Answer'].map((step, index) => (
            <span key={step} className="pipeline-step"><b>{String(index + 1).padStart(2, '0')}</b>{step}</span>
          ))}
        </div>
      </section>
    </div>
  )
}
