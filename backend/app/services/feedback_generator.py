from typing import Dict, Any, List
import numpy as np


class FeedbackGenerator:
    """Generate quarterback feedback based on video analysis"""
    
    def __init__(self):
        """Initialize feedback generator"""
        # Feedback thresholds
        self.thresholds = {
            'excellent': 80,
            'good': 60,
            'average': 40,
            'poor': 20
        }
    
    def generate(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive feedback for quarterback
        
        Args:
            analysis_results: Results from video processing
            
        Returns:
            Dictionary containing feedback and recommendations
        """
        openscore_summary = analysis_results.get('openscore_summary', {})
        tracking_summary = analysis_results.get('tracking_summary', {})
        
        # Analyze overall performance
        overall_analysis = self._analyze_overall_performance(openscore_summary)
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(openscore_summary, overall_analysis)
        
        # Identify best and worst decisions
        decision_analysis = self._analyze_decisions(openscore_summary)
        
        # Generate play-by-play insights
        key_moments = self._identify_key_moments(analysis_results.get('frame_data', []))
        
        return {
            'overall_grade': overall_analysis['grade'],
            'overall_score': overall_analysis['score'],
            'summary': overall_analysis['summary'],
            'strengths': overall_analysis['strengths'],
            'areas_for_improvement': overall_analysis['weaknesses'],
            'recommendations': recommendations,
            'best_options': decision_analysis['best_options'],
            'missed_opportunities': decision_analysis['missed_opportunities'],
            'key_moments': key_moments,
            'statistics': {
                'total_receivers_tracked': len(openscore_summary),
                'avg_openscore_all_receivers': overall_analysis['avg_score'],
                'total_players_detected': tracking_summary.get('total_tracks', 0)
            }
        }
    
    def _analyze_overall_performance(self, openscore_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall quarterback decision-making performance"""
        if not openscore_summary:
            return {
                'grade': 'N/A',
                'score': 0,
                'summary': 'No receiver data available for analysis',
                'strengths': [],
                'weaknesses': ['Insufficient data for analysis'],
                'avg_score': 0
            }
        
        # Collect all average openscores
        avg_scores = [data['avg_openscore'] for data in openscore_summary.values()]
        overall_avg = np.mean(avg_scores)
        
        # Determine grade
        if overall_avg >= self.thresholds['excellent']:
            grade = 'A'
            summary = "Excellent decision-making with consistently open receivers"
        elif overall_avg >= self.thresholds['good']:
            grade = 'B'
            summary = "Good decision-making with several quality passing options"
        elif overall_avg >= self.thresholds['average']:
            grade = 'C'
            summary = "Average decision-making with moderate passing opportunities"
        elif overall_avg >= self.thresholds['poor']:
            grade = 'D'
            summary = "Below average decision-making, receivers often covered"
        else:
            grade = 'F'
            summary = "Poor decision-making, limited open passing options"
        
        # Identify strengths
        strengths = []
        weaknesses = []
        
        # Analyze receiver openness distribution
        very_open_count = sum(1 for score in avg_scores if score >= 70)
        covered_count = sum(1 for score in avg_scores if score < 40)
        
        if very_open_count > len(avg_scores) * 0.5:
            strengths.append("Multiple receivers getting separation from defenders")
        
        if covered_count > len(avg_scores) * 0.5:
            weaknesses.append("Majority of receivers struggling to get open")
        
        # Analyze score variance
        score_std = np.std(avg_scores)
        if score_std < 15:
            strengths.append("Consistent receiver performance across all options")
        elif score_std > 30:
            weaknesses.append("High variance in receiver openness - need better read progression")
        
        # Check for elite performances
        max_score = max(avg_scores)
        if max_score >= 85:
            strengths.append(f"At least one receiver consistently wide open (OpenScore: {max_score:.1f})")
        
        return {
            'grade': grade,
            'score': round(overall_avg, 1),
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'avg_score': round(overall_avg, 1)
        }
    
    def _generate_recommendations(
        self,
        openscore_summary: Dict[str, Any],
        overall_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate specific recommendations for improvement"""
        recommendations = []
        
        avg_score = overall_analysis['avg_score']
        
        if avg_score < 50:
            recommendations.append(
                "Focus on reading defensive coverage pre-snap to identify potential openings"
            )
            recommendations.append(
                "Work with receivers on creating separation earlier in routes"
            )
        
        # Analyze specific receiver patterns
        if openscore_summary:
            scores = [data['avg_openscore'] for data in openscore_summary.values()]
            max_score = max(scores)
            min_score = min(scores)
            
            if max_score - min_score > 40:
                recommendations.append(
                    "Large variance in receiver openness detected - prioritize reads to most open receivers"
                )
            
            # Check for consistency
            for player_id, data in openscore_summary.items():
                if data['std_openscore'] > 25:
                    recommendations.append(
                        f"Receiver {player_id.replace('player_', '')} shows inconsistent separation - "
                        "timing and route adjustments may help"
                    )
                    break  # Only show one example
        
        if avg_score >= 60 and avg_score < 80:
            recommendations.append(
                "Good foundation - focus on exploiting highest OpenScore options earlier in progressions"
            )
        
        if avg_score >= 80:
            recommendations.append(
                "Excellent receiver separation - maintain timing and trust your reads"
            )
        
        # Always add general tip
        recommendations.append(
            "Continue analyzing game film to recognize coverage schemes faster"
        )
        
        return recommendations
    
    def _analyze_decisions(self, openscore_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Identify best options and missed opportunities"""
        best_options = []
        missed_opportunities = []
        
        if not openscore_summary:
            return {
                'best_options': best_options,
                'missed_opportunities': missed_opportunities
            }
        
        # Sort receivers by average openscore
        sorted_receivers = sorted(
            openscore_summary.items(),
            key=lambda x: x[1]['avg_openscore'],
            reverse=True
        )
        
        # Identify top 3 best options
        for i, (player_id, data) in enumerate(sorted_receivers[:3]):
            player_name = f"Receiver {player_id.replace('player_', '')}"
            best_options.append({
                'receiver': player_name,
                'avg_openscore': round(data['avg_openscore'], 1),
                'max_openscore': round(data['max_openscore'], 1),
                'consistency': 'High' if data['std_openscore'] < 15 else 'Moderate' if data['std_openscore'] < 25 else 'Low'
            })
        
        # Identify missed opportunities (receivers that had high peaks but low averages)
        for player_id, data in openscore_summary.items():
            if data['max_openscore'] >= 75 and data['avg_openscore'] < 55:
                player_name = f"Receiver {player_id.replace('player_', '')}"
                missed_opportunities.append({
                    'receiver': player_name,
                    'peak_openscore': round(data['max_openscore'], 1),
                    'avg_openscore': round(data['avg_openscore'], 1),
                    'note': 'Had moments of excellent separation but was covered most of the time'
                })
        
        return {
            'best_options': best_options,
            'missed_opportunities': missed_opportunities
        }
    
    def _identify_key_moments(self, frame_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key moments in the play"""
        key_moments = []
        
        if not frame_data:
            return key_moments
        
        # Find frames with highest openscores
        for i, frame in enumerate(frame_data):
            openscores = frame.get('openscores', {})
            
            if openscores:
                max_score = max(openscores.values())
                
                # Highlight exceptional moments
                if max_score >= 80:
                    max_receiver = max(openscores, key=openscores.get)
                    key_moments.append({
                        'frame': frame['frame_id'],
                        'type': 'excellent_opportunity',
                        'description': f"Receiver {max_receiver} wide open (OpenScore: {max_score:.1f})",
                        'openscore': round(max_score, 1)
                    })
        
        # Sort by openscore and limit to top 5
        key_moments.sort(key=lambda x: x['openscore'], reverse=True)
        
        return key_moments[:5]
