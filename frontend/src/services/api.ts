export interface AgentResponse {
  question: string
  answer: string
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')

export async function askAgent(question: string): Promise<AgentResponse> {
  let response: Response
  try {
    response = await fetch(`${apiBaseUrl}/api/agent/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new Error('Unable to connect to the HRMS AI backend. Make sure the FastAPI server is running.')
  }

  if (!response.ok) {
    if (response.status === 422) {
      throw new Error('Please enter a valid HR question.')
    }
    throw new Error('The HRMS AI backend could not process that question.')
  }

  return response.json() as Promise<AgentResponse>
}
