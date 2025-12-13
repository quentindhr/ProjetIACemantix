import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

interface StartResp {
  message: string;
  game_id: string;
  max_attempts: number;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private apiUrl = 'http://localhost:8000';
  constructor(private http: HttpClient) {}
  startGame(target?: string, max_attempts?: number): Observable<StartResp> {
    return this.http.post<StartResp>(`${this.apiUrl}/start`, { target, max_attempts });
  }
  guess(game_id: string, guess: string): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/guess`, { game_id, guess });
  }
  getVocab(limit: number = 200): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/vocab?limit=${limit}`);
  }
  aiSolve(game_id: string, useLLM: boolean = false, llmModel: string = 'ollama'): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/ai/solve`, { 
      game_id, 
      use_llm: useLLM,
      llm_model: llmModel
    });
  }
  getGameStatus(game_id: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/game/${game_id}`);
  }
}