import { Component, OnInit } from '@angular/core';
import { IdeaService, Idea, CheckResponse, MatchResult } from './idea.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  titleText = '';
  descriptionText = '';
  
  // State for loaded database ideas
  ideas: Idea[] = [];
  filteredIdeas: Idea[] = [];
  searchQuery = '';
  selectedTopic = '';
  allTopics: string[] = [];
  expandedIdeaId: number | null | undefined = null;

  // State for RAG duplicate check scanner
  isScanning = false;
  scanStep = 0;
  scanMessage = '';
  checkResult: CheckResponse | null = null;
  scannedIdea: { title: string; description: string } | null = null;
  activeWarnings: any[] = [];
  isLoading = false;
  isSaving = false;

  // Active status counters
  totalIdeasCount = 0;
  duplicatesBlockedCount = 0;

  constructor(private ideaService: IdeaService) {}

  ngOnInit() {
    this.loadIdeas(true);
  }

  loadIdeas(isBoot = false) {
    if (isBoot) {
      this.isLoading = true;
    }
    this.ideaService.getIdeas().subscribe({
      next: (data) => {
        this.ideas = data;
        this.filteredIdeas = [...data];
        this.totalIdeasCount = data.length;
        
        // Compile a list of unique topics/categories across all ideas
        const topicsSet = new Set<string>();
        data.forEach(idea => {
          if (idea.topics) {
            idea.topics.forEach(t => topicsSet.add(t));
          }
        });
        this.allTopics = Array.from(topicsSet);
        this.applyFilter();
        this.isLoading = false;
      },
      error: (err) => {
        console.error('Error fetching ideas:', err);
        this.isLoading = false;
      }
    });
  }

  /**
   * Triggers the animated scanner and calls the backend duplicate check endpoint.
   */
  runScan() {
    if (!this.titleText.trim() || !this.descriptionText.trim()) return;

    this.isScanning = true;
    this.scanStep = 1;
    this.scanMessage = 'Initializing AI Analysis Engines...';
    this.checkResult = null;
    this.scannedIdea = { title: this.titleText, description: this.descriptionText };

    // Trigger step-by-step scanner animation milestones to wow the user
    setTimeout(() => {
      this.scanStep = 2;
      this.scanMessage = 'Generating Dense Vector Embeddings (Gemini gemini-embedding-2)...';
    }, 1000);

    setTimeout(() => {
      this.scanStep = 3;
      this.scanMessage = 'Executing Qdrant Hybrid Search (Sparse BM25 + Dense Cosine Similarity)...';
    }, 2000);

    setTimeout(() => {
      this.scanStep = 4;
      this.scanMessage = 'Running Cross-Encoder Reranker model locally...';
    }, 3000);

    // Call API check endpoint
    this.ideaService.checkDuplicate({ title: this.titleText, description: this.descriptionText }).subscribe({
      next: (res) => {
        // Complete the animation timeline before showing result
        setTimeout(() => {
          this.isScanning = false;
          this.checkResult = res;
          if (res.is_duplicate) {
            this.duplicatesBlockedCount++;
          }
          if (res.warnings && res.warnings.length > 0) {
            this.activeWarnings = [...this.activeWarnings, ...res.warnings];
          }
        }, 3800);
      },
      error: (err) => {
        console.error('Scan error:', err);
        this.isScanning = false;
        alert('An error occurred during scanning. Make sure the backend server is running.');
      }
    });
  }

  /**
   * Saves the submitted idea to the database after validating it.
   */
  submitIdea() {
    if (!this.titleText.trim() || !this.descriptionText.trim()) return;

    this.isSaving = true;

    // Call submit API
    this.ideaService.createIdea({ title: this.titleText, description: this.descriptionText }).subscribe({
      next: (savedIdea) => {
        this.isSaving = false;
        // Reset form and reload list
        this.titleText = '';
        this.descriptionText = '';
        this.checkResult = null;
        this.scannedIdea = null;
        if (savedIdea.warnings && savedIdea.warnings.length > 0) {
          this.activeWarnings = [...this.activeWarnings, ...savedIdea.warnings];
        }
        this.loadIdeas();
      },
      error: (err) => {
        console.error('Submission error:', err);
        this.isSaving = false;
        alert('Error saving the idea.');
      }
    });
  }

  /**
   * Resets the scan results board and form fields.
   */
  clearScan() {
    this.checkResult = null;
    this.scannedIdea = null;
  }

  dismissWarning(idx: number) {
    this.activeWarnings.splice(idx, 1);
  }

  /**
   * Deletes an idea from the database and refreshes the repository list.
   */
  deleteIdea(ideaId: number | undefined, event: MouseEvent) {
    if (!ideaId) return;
    
    // Stop propagation so clicking delete doesn't expand/collapse the card
    event.stopPropagation();
    
    this.ideaService.deleteIdea(ideaId).subscribe({
      next: () => {
        this.loadIdeas();
      },
      error: (err) => {
        console.error('Delete error:', err);
        alert('Failed to delete the idea.');
      }
    });
  }

  /**
   * Applies keyword searches and topic filters to the gallery checklist.
   */
  applyFilter() {
    this.filteredIdeas = this.ideas.filter(idea => {
      // Filter by search query (checks Title, Description, and Tags)
      const matchesSearch = !this.searchQuery.trim() || 
        idea.title.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
        idea.description.toLowerCase().includes(this.searchQuery.toLowerCase()) ||
        (idea.tags && idea.tags.some(tag => tag.toLowerCase().includes(this.searchQuery.toLowerCase())));

      // Filter by category tag click
      const matchesTopic = !this.selectedTopic || 
        (idea.topics && idea.topics.includes(this.selectedTopic));

      return matchesSearch && matchesTopic;
    });
  }

  /**
   * Filters the ideas by clicking a specific topic badge.
   */
  filterByTopic(topic: string) {
    if (this.selectedTopic === topic) {
      this.selectedTopic = ''; // Toggle filter off if clicked again
    } else {
      this.selectedTopic = topic;
    }
    this.applyFilter();
  }

  /**
   * Toggles the detail view of an idea card.
   */
  toggleExpand(ideaId: number | undefined) {
    if (this.expandedIdeaId === ideaId) {
      this.expandedIdeaId = null;
    } else {
      this.expandedIdeaId = ideaId;
    }
  }

  /**
   * Computes clean styling tags based on similarity percentage.
   */
  getScoreClass(score: number): string {
    if (score >= 0.80) return 'high-danger';
    if (score >= 0.50) return 'medium-warning';
    return 'low-safe';
  }

  /**
   * Returns rounded similarity percentages for easy reading.
   */
  getPercent(score: number): number {
    return Math.round(score * 100);
  }
}
