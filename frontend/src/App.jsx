import { useState, useEffect } from 'react';
import { Send, Download, Trash2, RefreshCw } from 'lucide-react';

export default function CodeAnalyzer() {
  const [codeInput, setCodeInput] = useState('');
  const [analyses, setAnalyses] = useState([]);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Get API URL from environment variables
  // const API_URL = (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_URL) || 'http://localhost:8000/api';
  const API_URL = import.meta.env.VITE_API_URL;

  // Fetch all analyses on mount
  useEffect(() => {
    fetchAnalyses();
  }, []);

  const fetchAnalyses = async () => {
    try {
      const res = await fetch(`${API_URL}/api/analyses?limit=50`);
      if (!res.ok) throw new Error('Failed to fetch analyses');
      const data = await res.json();
      setAnalyses(data);
    } catch (err) {
      setError('Failed to load analyses');
      console.error(err);
    }
  };

  const handleAnalyze = async () => {
    if (!codeInput.trim()) {
      setError('Please enter code to analyze');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code_snippet: codeInput })
      });

      if (!res.ok) throw new Error('Analysis failed');

      const result = await res.json();
      setSelectedAnalysis(result);
      setAnalyses([result, ...analyses]);
      setCodeInput('');
    } catch (err) {
      setError('Failed to analyze code');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this analysis?')) return;

    try {
      const res = await fetch(`${API_URL}/api/analysis/${id}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Delete failed');

      setAnalyses(analyses.filter(a => a.id !== id));
      if (selectedAnalysis?.id === id) setSelectedAnalysis(null);
    } catch (err) {
      setError('Failed to delete analysis');
    }
  };

  const downloadResult = (analysis) => {
    const content = `Code Analysis Report
====================
Date: ${new Date(analysis.created_at).toLocaleString()}
ID: ${analysis.id}

ORIGINAL CODE:
${analysis.original_code}

FIXED CODE:
${analysis.fixed_code}

Risk SCORES:
Before: ${analysis.before_score}/100
After: ${analysis.after_score}/100
Improvement: ${(analysis.before_score - analysis.after_score).toFixed(2)} points
`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${analysis.id}.txt`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const ScoreDisplay = ({ before, after }) => {
    const improvement = before.toFixed(1) - after.toFixed(1);
    return (
      <div className="flex gap-6">
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-2">Before</div>
          <div className="text-3xl font-bold text-red-500">{before.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Risk Score</div>
        </div>
        <div className="flex items-center justify-center">
          <div className="text-2xl text-gray-400">→</div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-2">After</div>
          <div className="text-3xl font-bold text-green-500">{after.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Risk Score</div>
        </div>
        <div className="flex items-center justify-center">
          <div className={`text-lg font-bold ${improvement >= 0 ? 'text-green-500' : 'text-red-500'}`}>
              {improvement >= 0 ? '+' : ''} {improvement.toFixed(1)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-white mb-2">Code Security Analyzer</h1>
        <p className="text-gray-400 mb-8">Analyze and secure your code with AI-powered insights</p>

        <div className="grid grid-cols-3 gap-6">
          {/* Input Section */}
          <div className="col-span-2 space-y-4">
            <div className="bg-slate-700 rounded-lg p-6">
              <label className="block text-sm font-semibold text-gray-200 mb-3">
                Code Snippet
              </label>
              <textarea
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                placeholder="Paste your code here..."
                className="w-full h-64 bg-slate-800 text-gray-100 border border-slate-600 rounded-lg p-4 font-mono text-sm focus:outline-none focus:border-blue-500 resize-none"
              />
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 text-white font-semibold py-3 rounded-lg flex items-center justify-center gap-2 transition"
              >
                {loading ? <RefreshCw className="animate-spin" size={20} /> : <Send size={20} />}
                {loading ? 'Analyzing...' : 'Analyze Code'}
              </button>
            </div>

            {/* Results Section */}
            {selectedAnalysis && (
              <div className="bg-slate-700 rounded-lg p-6 space-y-6">
                <div>
                  <h2 className="text-xl font-bold text-white mb-4">Risk Scores</h2>
                  <ScoreDisplay before={selectedAnalysis.before_score} after={selectedAnalysis.after_score} />
                </div>

                <div className="grid grid-cols-1 gap-4">
                  <div>
                    <h3 className="text-sm font-semibold text-gray-300 mb-2">Fixed Code</h3>
                    <pre className="bg-slate-800 p-3 rounded text-xs text-gray-100 overflow-auto max-h-40 font-mono">
                      {selectedAnalysis.fixed_code}
                    </pre>
                  </div>
                </div>

                <button
                  onClick={() => downloadResult(selectedAnalysis)}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded-lg flex items-center justify-center gap-2 transition"
                >
                  <Download size={18} /> Download Report
                </button>
              </div>
            )}

            {error && (
              <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}
          </div>

          {/* History Sidebar */}
          <div className="bg-slate-700 rounded-lg p-6 h-fit sticky top-6">
            <h2 className="text-lg font-bold text-white mb-4">Analysis History</h2>
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {analyses.length === 0 ? (
                <p className="text-gray-400 text-sm">No analyses yet</p>
              ) : (
                analyses.map((analysis) => (
                  <div
                    key={analysis.id}
                    onClick={() => setSelectedAnalysis(analysis)}
                    className={`p-3 rounded-lg cursor-pointer transition ${
                      selectedAnalysis?.id === analysis.id
                        ? 'bg-blue-600'
                        : 'bg-slate-600 hover:bg-slate-500'
                    }`}
                  >
                    <div className="text-sm text-gray-100 truncate font-mono">
                      {analysis.original_code.substring(0, 20)}...
                    </div>
                    <div className="text-xs text-gray-300 mt-1">
                      {new Date(analysis.created_at).toLocaleDateString()}
                    </div>
                    <div className="text-xs text-green-400 font-semibold mt-1">
                      Score: {analysis.after_score.toFixed(1)}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(analysis.id);
                      }}
                      className="mt-2 w-full bg-red-600 hover:bg-red-700 text-white text-xs py-1 rounded flex items-center justify-center gap-1"
                    >
                      <Trash2 size={12} /> Delete
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}