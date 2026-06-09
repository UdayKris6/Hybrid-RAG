import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Idea {
  id?: number;
  title: string;
  description: string;
  summary?: string;
  topics?: string[];
  tags?: string[];
  status?: string;
  created_at?: string;
  warnings?: any[];
}

export interface MatchResult {
  id: number;
  score: number;
  similarity_score: number;
  rrf_score: number;
  title: string;
  description: string;
  summary: string;
  topics: string[];
  tags: string[];
}

export interface CheckResponse {
  is_duplicate: boolean;
  max_similarity_score: number;
  matches: MatchResult[];
  warnings?: any[];
}

@Injectable({
  providedIn: 'root'
})
export class IdeaService {
  private apiUrl = 'http://localhost:8000/api/ideas';

  constructor(private http: HttpClient) { }

  /**
   * Retrieves all submitted ideas from the backend database.
   */
  getIdeas(): Observable<Idea[]> {
    return this.http.get<Idea[]>(this.apiUrl);
  }

  /**
   * Submits and saves a new idea in the system, extracting its tags and vectors.
   */
  createIdea(idea: { title: string; description: string }): Observable<Idea> {
    return this.http.post<Idea>(this.apiUrl, idea);
  }

  /**
   * Runs the draft idea through the Hybrid Search and Reranker without saving it.
   */
  checkDuplicate(idea: { title: string; description: string }): Observable<CheckResponse> {
    return this.http.post<CheckResponse>(`${this.apiUrl}/check`, idea);
  }
}
