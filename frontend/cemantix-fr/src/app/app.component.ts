import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from './api.service';


@Component({
  standalone: true,
  imports: [CommonModule, FormsModule],
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  gameId: string | null = null;
  maxAttempts = 6;
  guessText = '';
  history: any[] = [];
  remaining = 0;
  finished = false;
  won = false;
  targetWord: string | null = null;
  message = '';
  topSimilaires: any[] = [];
  aiSolving = false;

  constructor(private api: ApiService) {}

  newGame() {
    this.api.startGame(undefined, this.maxAttempts).subscribe(res => {
      this.gameId = res.game_id;
      this.history = [];
      this.remaining = res.max_attempts;
      this.finished = false;
      this.won = false;
      this.targetWord = null;
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
      // Mapper l'historique avec les informations
      const historyMapped = res.history.map((h: any, index: number) => ({ 
        guess: h.guess, 
        score: h.score, // Le score est déjà en pourcentage (0-100)
        rank: h.rank || res.rank, // Utiliser le rang de l'historique ou celui du dernier guess
        attempt: index + 1
      }));
      
      // Trier par score décroissant
      this.history = historyMapped.sort((a: any, b: any) => b.score - a.score);
      
      this.remaining = res.remaining;
      this.finished = res.finished;
      this.won = res.won;
      this.targetWord = res.target || null;
      this.topSimilaires = res.top_similaires || [];
      
      if (res.finished) {
        this.message = res.won ? `🎉 Bravo ! Vous avez trouvé le mot !` : `😔 Partie terminée`;
      } else {
        this.message = `Score: ${res.score.toFixed(1)}% — Rang: ${res.rank}`;
      }
      this.guessText = '';
    }, err => {
      this.message = 'Erreur lors de la proposition : ' + (err.error?.detail ?? err.message);
    });
  }

  // Calculate percentage from score (assuming score is between 0 and 1)
  getProximityPercentage(score: number): number {
    if (score < 0) return 0;
    if (score > 100) return 100;
    return score
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

  aiSolve() {
    if (!this.gameId) {
      this.message = "D'abord démarrer une partie";
      return;
    }
    if (this.finished) {
      this.message = "La partie est déjà terminée";
      return;
    }
    
    this.aiSolving = true;
    this.message = '🤖 L\'IA résout la partie...';
    
    this.api.aiSolve(this.gameId).subscribe(res => {
      this.aiSolving = false;
      
      if (res.success) {
        this.message = `🤖 L'IA a trouvé le mot en ${res.attempts} essai(s) !`;
        // Rafraîchir l'état de la partie
        this.refreshGameState();
      } else {
        if (res.error) {
          this.message = `❌ Erreur IA: ${res.error}`;
        } else {
          this.message = `🤖 L'IA n'a pas trouvé le mot en ${res.attempts} essai(s)`;
          this.refreshGameState();
        }
      }
    }, err => {
      this.aiSolving = false;
      this.message = 'Erreur lors de la résolution IA : ' + (err.error?.detail ?? err.message);
    });
  }

  refreshGameState() {
    if (!this.gameId) return;
    
    // Récupérer l'état actuel de la partie
    this.api.getGameStatus(this.gameId).subscribe(res => {
      const historyMapped = res.history.map((h: any, index: number) => ({ 
        guess: h.guess, 
        score: h.score,
        rank: h.rank,
        attempt: index + 1
      }));
      
      this.history = historyMapped.sort((a: any, b: any) => b.score - a.score);
      this.remaining = res.max_attempts - res.attempts;
      this.finished = res.finished;
      this.won = res.won;
      this.targetWord = res.target || null;
    }, err => {
      console.error('Erreur lors du rafraîchissement:', err);
    });
  }
}