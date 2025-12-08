import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';


@Component({
  standalone: true,
  imports: [CommonModule, FormsModule],
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  gameId: string | null = null;
  maxAttempts = 6;
  guessText = '';
  history: any[] = [];
  remaining = 0;
  finished = false;
  won = false;
  message = '';
  topSimilaires: any[] = [];

  constructor(private api: ApiService) {}

  newGame() {
    this.api.startGame(undefined, this.maxAttempts).subscribe(res => {
      this.gameId = res.game_id;
      this.history = [];
      this.remaining = res.max_attempts;
      this.finished = false;
      this.won = false;
      this.message = 'Nouvelle partie démarrée. Bonne chance !';
    }, err => {
      this.message = 'Erreur démarrage partie : ' + (err.error?.detail ?? err.message);
    });
  }

  submitGuess() {
    if (!this.gameId) {
      this.message = "D'abord démarrer une partie";
      return;
    }
    const guess = this.guessText.trim();
    if (!guess) {
      this.message = "Entrez un mot.";
      return;
    }
    this.api.guess(this.gameId, guess).subscribe(res => {
      if (res.error) {
        this.message = res.error;
        return;
      }
      this.history = res.history.map((h: any) => ({ guess: h.guess, score: Math.round(h.score * 1000) / 1000 }))
        .sort((a: any, b: any) => b.score - a.score); // Sort by score descending
      this.remaining = res.remaining;
      this.finished = res.finished;
      this.won = res.won;
      this.topSimilaires = res.top_similaires || [];
      if (res.finished) {
        this.message = res.won ? `Bravo ! Vous avez trouvé : ${res.target}` : `Partie terminée. Le mot était : ${res.target}`;
      } else {
        this.message = `Score: ${Math.round(res.score * 1000) / 1000} — Rang approximatif: ${res.rank}`;
      }
      this.guessText = '';
    }, err => {
      this.message = 'Erreur lors de la proposition : ' + (err.error?.detail ?? err.message);
    });
  }

  // Calculate percentage from score (assuming score is between 0 and 1)
  getProximityPercentage(score: number): number {
    return Math.max(0, Math.min(100, score * 100));
  }

  // Determine gauge color based on score
  getGaugeColor(score: number): string {
    const percentage = this.getProximityPercentage(score);
    if (percentage >= 80) return '#4CAF50'; // Green
    if (percentage >= 60) return '#8BC34A'; // Yellow-green
    if (percentage >= 40) return '#FF9800'; // Orange
    if (percentage >= 20) return '#FF5722'; // Red
    return '#D32F2F'; // Dark red
  }

  // Check if score is very high (for pulse animation)
  isVeryClose(score: number): boolean {
    return this.getProximityPercentage(score) > 90;
  }
}