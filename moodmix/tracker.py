import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from textblob import TextBlob
import pandas as pd
import plotly.express as px

@dataclass
class MoodEntry:
    timestamp: datetime.datetime
    mood_text: str
    mood_score: float
    activities: List[str]
    sentiment_score: float

class MoodTracker:
    def __init__(self):
        self.mood_entries: List[MoodEntry] = []
    
    def add_entry(self, mood_text: str, activities: List[str]) -> MoodEntry:
        """Add a new mood entry with automatic sentiment analysis"""
        # Analyze sentiment using TextBlob
        blob = TextBlob(mood_text)
        sentiment_score = blob.sentiment.polarity
        
        # Map sentiment to mood score (0-10 scale)
        mood_score = (sentiment_score + 1) * 5
        
        entry = MoodEntry(
            timestamp=datetime.datetime.now(),
            mood_text=mood_text,
            mood_score=mood_score,
            activities=activities,
            sentiment_score=sentiment_score
        )
        
        self.mood_entries.append(entry)
        return entry
    
    def get_mood_trends(self, days: int = 7) -> Dict:
        """Analyze mood trends over specified number of days"""
        if not self.mood_entries:
            return {}
            
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_entries = [e for e in self.mood_entries if e.timestamp >= cutoff]
        
        if not recent_entries:
            return {}
        
        df = pd.DataFrame([
            {
                'date': e.timestamp,
                'mood_score': e.mood_score,
                'activities': ','.join(e.activities)
            } for e in recent_entries
        ])
        
        # Calculate statistics
        trends = {
            'average_mood': df['mood_score'].mean(),
            'mood_variance': df['mood_score'].var(),
            'best_day': df.loc[df['mood_score'].idxmax()]['date'].strftime('%Y-%m-%d'),
            'worst_day': df.loc[df['mood_score'].idxmin()]['date'].strftime('%Y-%m-%d'),
            'common_activities': self._get_common_activities(recent_entries)
        }
        
        return trends
    
    def generate_mood_graph(self, days: int = 7) -> Optional[str]:
        """Generate an interactive mood trend visualization"""
        if not self.mood_entries:
            return None
            
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        recent_entries = [e for e in self.mood_entries if e.timestamp >= cutoff]
        
        if not recent_entries:
            return None
        
        df = pd.DataFrame([
            {
                'date': e.timestamp,
                'mood_score': e.mood_score,
                'activities': ','.join(e.activities)
            } for e in recent_entries
        ])
        
        fig = px.line(
            df,
            x='date',
            y='mood_score',
            title='Your Mood Trends',
            labels={'date': 'Date', 'mood_score': 'Mood Score (0-10)'},
            markers=True
        )
        
        return fig.to_html()
    
    def _get_common_activities(self, entries: List[MoodEntry]) -> List[str]:
        """Helper method to find most common activities"""
        activity_count = {}
        for entry in entries:
            for activity in entry.activities:
                activity_count[activity] = activity_count.get(activity, 0) + 1
                
        # Sort by frequency and return top 5
        sorted_activities = sorted(
            activity_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [activity for activity, _ in sorted_activities[:5]]
