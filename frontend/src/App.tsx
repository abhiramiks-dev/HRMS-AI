import { useState } from 'react'
import { ArchitecturePanel } from './components/ArchitecturePanel'
import { AnswerPanel } from './components/AnswerPanel'
import { Header } from './components/Header'
import { QuestionInput } from './components/QuestionInput'
import { askAgent } from './services/api'
import './styles.css'

export default function App() {
  const [question, setQuestion] = useState('')
  const [submittedQuestion, setSubmittedQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit() {
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || loading) return

    setLoading(true)
    setError(null)
    setAnswer('')
    setSubmittedQuestion(trimmedQuestion)
    try {
      const response = await askAgent(trimmedQuestion)
      setAnswer(response.answer)
    } catch (requestError) {
      setError(requestError instanceof Error
        ? requestError.message
        : 'Unable to connect to the HRMS AI backend. Make sure the FastAPI server is running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <Header />
      <main>
        <QuestionInput question={question} loading={loading} onChange={setQuestion} onSubmit={handleSubmit} />
        <AnswerPanel question={submittedQuestion} answer={answer} loading={loading} error={error} />
        <ArchitecturePanel />
      </main>
      <footer>Built with FastAPI · React · semantic retrieval · Gemini</footer>
    </div>
  )
}
