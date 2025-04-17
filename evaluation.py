#Name : Nathaniel Monney
#Index Nunber : 10211100300



import pandas as pd
import re
import json
import os
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score

class RagEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        self.eval_results = []
        self.gpt_comparison = []

    def evaluate_response(self, question, context, response, ground_truth=None):
        """Evaluate a single RAG response"""
        eval_result = {
            "question": question,
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {}
        }

        # Extract keywords
        keywords = self._extract_keywords(question)

        # Context relevance
        context_relevance = sum(1 for kw in keywords if kw.lower() in context.lower()) / max(1, len(keywords))
        eval_result["metrics"]["context_relevance"] = context_relevance

        # Response completeness
        response_completeness = sum(1 for kw in keywords if kw.lower() in response.lower()) / max(1, len(keywords))
        eval_result["metrics"]["response_completeness"] = response_completeness

        # Response conciseness
        response_conciseness = min(1.0, len(context) / max(1, len(response)))
        eval_result["metrics"]["response_conciseness"] = response_conciseness

        # Ground truth comparison (simulated precision, recall, f1)
        if ground_truth:
            eval_result["ground_truth"] = ground_truth
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            truth_words = set(re.findall(r'\b\w+\b', ground_truth.lower()))
            if truth_words:
                # Simulated binary classification for words
                all_words = list(set(response_words).union(truth_words))
                y_true = [1 if w in truth_words else 0 for w in all_words]
                y_pred = [1 if w in response_words else 0 for w in all_words]
                eval_result["metrics"]["precision"] = precision_score(y_true, y_pred, zero_division=0)
                eval_result["metrics"]["recall"] = recall_score(y_true, y_pred, zero_division=0)
                eval_result["metrics"]["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

        self.eval_results.append(eval_result)
        return eval_result

    def evaluate_multimodal_response(self, question, context, response, images=None):
        """Evaluate a multimodal response"""
        eval_result = self.evaluate_response(question, context, response)
        eval_result["metrics"]["visual_reference"] = 0.0
        eval_result["metrics"]["visual_analysis"] = 0.0

        if images:
            # Check if response references visual elements
            visual_keywords = ['chart', 'graph', 'image', 'visual', 'figure', 'diagram']
            eval_result["metrics"]["visual_reference"] = 1.0 if any(kw in response.lower() for kw in visual_keywords) else 0.0
            # Check if response analyzes visual content (basic heuristic)
            eval_result["metrics"]["visual_analysis"] = 1.0 if len(response) > 100 and eval_result["metrics"]["visual_reference"] > 0 else 0.0

        return eval_result

    def compare_with_chatgpt(self, question, rag_response, chatgpt_response):
        """Compare RAG response with ChatGPT response"""
        comparison = {
            "question": question,
            "rag_response": rag_response,
            "chatgpt_response": chatgpt_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "comparison": {}
        }

        rag_words = set(re.findall(r'\b\w+\b', rag_response.lower()))
        gpt_words = set(re.findall(r'\b\w+\b', chatgpt_response.lower()))

        # Jaccard similarity
        union = len(rag_words.union(gpt_words))
        intersection = len(rag_words.intersection(gpt_words))
        similarity = intersection / max(1, union)
        comparison["comparison"]["response_similarity"] = similarity

        # Length comparison
        rag_length = len(rag_response.split())
        gpt_length = len(chatgpt_response.split())
        length_ratio = min(rag_length, gpt_length) / max(1, max(rag_length, gpt_length))
        comparison["comparison"]["length_ratio"] = length_ratio

        self.gpt_comparison.append(comparison)
        return comparison

    def _extract_keywords(self, text):
        """Extract potential keywords from the question"""
        stopwords = ["a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of"]
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        return keywords

    def save_evaluations(self, directory="./evaluations"):
        """Save evaluation results to JSON files"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.eval_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{directory}/rag_eval_{timestamp}.json", 'w') as f:
                json.dump(self.eval_results, f, indent=2)

        if self.gpt_comparison:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{directory}/gpt_comparison_{timestamp}.json", 'w') as f:
                json.dump(self.gpt_comparison, f, indent=2)

    def generate_summary(self):
        """Generate a summary of all evaluations"""
        summary = {
            "total_evaluations": len(self.eval_results),
            "total_comparisons": len(self.gpt_comparison),
            "average_metrics": {},
            "comparison_averages": {}
        }

        if self.eval_results:
            metrics_keys = set()
            for result in self.eval_results:
                metrics_keys.update(result["metrics"].keys())

            for key in metrics_keys:
                values = [r["metrics"].get(key, 0) for r in self.eval_results if key in r["metrics"]]
                if values:
                    summary["average_metrics"][key] = sum(values) / len(values)

        if self.gpt_comparison:
            comparison_keys = set()
            for comp in self.gpt_comparison:
                comparison_keys.update(comp["comparison"].keys())

            for key in comparison_keys:
                values = [c["comparison"].get(key, 0) for c in self.gpt_comparison if key in c["comparison"]]
                if values:
                    summary["comparison_averages"][key] = sum(values) / len(values)

        return summary