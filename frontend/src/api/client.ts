const base = ''
async function request<T>(path:string, init?:RequestInit):Promise<T>{
  const r=await fetch(`${base}${path}`,init)
  const text=await r.text()
  let body:any={}
  try{body=text?JSON.parse(text):{}}catch{body={detail:text}}
  if(!r.ok) throw new Error(body?.detail || `Request failed (${r.status})`)
  return body as T
}
export const apiGet=<T,>(path:string)=>request<T>(path)
export const apiPostFile=<T,>(path:string,file:File,asOf=24,horizon=168)=>{
  const fd=new FormData(); fd.append('file',file)
  return request<T>(`${path}?as_of_h=${encodeURIComponent(asOf)}&target_horizon=${encodeURIComponent(horizon)}`,{method:'POST',body:fd})
}
