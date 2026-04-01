import collections
from typing import List, Dict, Optional

class Text2SQLTableRanker:
    def __init__(self):
        self.weight_desc = 0.6
        self.weight_project = 0.27
        self.weight_column = 0.13
        self.min_confidence = 0.38
        self.min_desc_score = 0.45
        self.low_desc_score = 0.30
        self.desc_high_threshold = 0.75
        self.project_synergy_bonus = 0.04
        self.column_synergy_bonus = 0.02
        self.min_score_gap = 0.03
        self.min_desc_gap = 0.05

    def parse_and_aggregate(self, raw_data: List[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
        temp_store = collections.defaultdict(lambda: collections.defaultdict(list))
        for item in raw_data:
            t_name = item['table']
            source = item['source']
            score = float(item['score'])
            temp_store[t_name][source].append(score)

        aggregated = {}
        for t_name, sources in temp_store.items():
            aggregated[t_name] = {}
            for source, scores in sources.items():
                aggregated[t_name][source] = {
                    'max_score': max(scores),
                    'count': len(scores)
                }
        return aggregated

    def count_to_strength(self, count: int) -> float:
        if count <= 0:
            return 0.0
        elif count == 1:
            return 0.65
        elif count == 2:
            return 0.78
        elif count == 3:
            return 0.88
        else:
            return 0.95

    def calculate_score(self, stats: Dict[str, Dict[str, float]]) -> float:
        desc_score = stats.get('表描述', {}).get('max_score', 0.0)
        project_count = stats.get('项目匹配', {}).get('count', 0)
        column_count = stats.get('列匹配', {}).get('count', 0)

        project_strength = self.count_to_strength(project_count)
        column_strength = self.count_to_strength(column_count)

        final_score = (
                self.weight_desc * desc_score +
                self.weight_project * project_strength +
                self.weight_column * column_strength
        )

        if desc_score >= self.desc_high_threshold and project_count > 0:
            final_score += self.project_synergy_bonus

        if desc_score >= self.desc_high_threshold and column_count > 0:
            final_score += self.column_synergy_bonus

        return final_score

    def rank(self, raw_data: List[Dict]) -> Optional[Dict]:
        candidates = self.parse_and_aggregate(raw_data)
        if not candidates:
            return None

        results = []

        for t_name, stats in candidates.items():
            final_score = self.calculate_score(stats)
            desc_score = stats.get('表描述', {}).get('max_score', 0.0)
            project_count = stats.get('项目匹配', {}).get('count', 0)
            column_count = stats.get('列匹配', {}).get('count', 0)

            score_breakdown = {
                '表描述分': desc_score,
                '项目匹配次数': project_count,
                '列匹配次数': column_count,
                '项目匹配强度': self.count_to_strength(project_count),
                '列匹配强度': self.count_to_strength(column_count),
            }

            results.append({
                'table': t_name,
                'final_score': round(final_score, 6),
                'details': score_breakdown
            })

        results.sort(key=lambda x: x['final_score'], reverse=True)
        best_match = results[0]

        if best_match['final_score'] < self.min_confidence:
            return {
                'status': 'LOW_CONFIDENCE',
                'message': f"最高得分 {best_match['final_score']:.4f} 低于总分阈值 {self.min_confidence}",
                'top_table': best_match['table'],
                'all_candidates': results
            }
        
        if len(results) > 1:
            if best_match['final_score'] < self.min_confidence:
                return {
                    'status': 'LOW_CONFIDENCE',
                    'message': f"最高得分 {best_match['final_score']:.4f} 低于阈值 {self.min_confidence}",
                    'top_table': best_match['table'],
                    'all_candidates': results
                }

        return {
            'status': 'SUCCESS',
            'selected_table': best_match['table'],
            'final_score': best_match['final_score'],
            'score_breakdown': best_match['details'],
            'all_candidates': results
        }