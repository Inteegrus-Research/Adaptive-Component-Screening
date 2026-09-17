import React, { ErrorInfo, ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { hasError: boolean; message: string }

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false, message: '' }
  static getDerivedStateFromError(error: Error): State { return { hasError: true, message: error.message } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('ACS UI error', error, info) }
  render() {
    if (!this.state.hasError) return this.props.children
    return <div className="fatalPage"><div className="fatalPanel"><div className="eyebrow">APPLICATION SAFETY STOP</div><h1>Interface recovered from an unexpected state</h1><p>{this.state.message || 'The UI encountered an unexpected rendering error.'}</p><button className="btn primary" onClick={() => window.location.reload()}>Reload console</button></div></div>
  }
}
