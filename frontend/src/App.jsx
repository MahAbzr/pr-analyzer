import React, { useState } from 'react';
import { Search, Code, AlertCircle, CheckCircle, Clock, Zap } from 'lucide-react';

export default function CodeAnalyzer() {
  const [inputType, setInputType] = useState('url');
  const [url, setUrl] = useState('');
  const [code, setCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [analysisId, setAnalysisId] = useState('');
  const [checkInterval, setCheckInterval] = useState(null);

  const API_URL = 'http://localhost:8000/api';

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const payload = inputType === 'url'
        ? { url }
        : { code };

      const response = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      setAnalysisId(data.id);
      setResults({
        id: data.id,
        status: 'processing',
        extracted_data: data.extracted_data,
        ml_result: null,
      });

      // Poll for results
      const interval = setInterval(() => pollResults(data.id), 2000);
      setCheckInterval(interval);
    } catch (err) {
      alert('Error: ' + err.message);
      setLoading(false);
    }
  };

  const pollResults = async (id) => {
    try {
      const response = await fetch(`${API_URL}/analysis/${id}`);
      if (!response.ok) throw new Error('Failed to fetch results');

      const data = await response.json();
      setResults(data);

      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(checkInterval);
        setLoading(false);
      }
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Code className="w-10 h-10 text-purple-400" />
            <h1 className="text-4xl font-bold text-white">Code Analyzer</h1>
          </div>
          <p className="text-gray-300">Analyze code quality, complexity, and get AI-powered insights</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Section */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800 rounded-lg p-6 border border-purple-500/20">
              <h2 className="text-xl font-semibold text-white mb-6">Input Code</h2>

              <div className="space-y-4">
                {/* Toggle Input Type */}
                <div className="flex gap-3 mb-6">
                  <button
                    onClick={() => setInputType('url')}
                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                      inputType === 'url'
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                    }`}
                  >
                    <Search className="w-4 h-4 inline mr-2" />
                    URL
                  </button>
                  <button
                    onClick={() => setInputType('code')}
                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                      inputType === 'code'
                        ? 'bg-purple-600 text-white'
                        : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                    }`}
                  >
                    <Code className="w-4 h-4 inline mr-2" />
                    Direct
                  </button>
                </div>

                {/* Input Field */}
                {inputType === 'url' ? (
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://github.com/user/repo/pull/123"
                    className="w-full px-4 py-3 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-purple-500 focus:outline-none"
                  />
                ) : (
                  <textarea
                    value={code}
                    onChange={(e) => setCode(e.target.value)}
                    placeholder="Paste your code here..."
                    rows="12"
                    className="w-full px-4 py-3 bg-slate-700 text-white rounded-lg border border-slate-600 focus:border-purple-500 focus:outline-none font-mono text-sm"
                  />
                )}

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={loading || (!url && !code)}
                  className="w-full py-3 px-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white font-semibold rounded-lg hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                >
                  {loading ? 'Analyzing...' : 'Analyze Code'}
                </button>
              </div>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {!results ? (
              <div className="bg-slate-800 rounded-lg p-12 border border-purple-500/20 text-center">
                <Code className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">Submit code to see analysis results</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Status Card */}
                <div className="bg-slate-800 rounded-lg p-6 border border-purple-500/20">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-semibold text-white">Analysis Status</h3>
                    {results.status === 'processing' && (
                      <Clock className="w-5 h-5 text-yellow-400 animate-spin" />
                    )}
                    {results.status === 'completed' && (
                      <CheckCircle className="w-5 h-5 text-green-400" />
                    )}
                    {results.status === 'failed' && (
                      <AlertCircle className="w-5 h-5 text-red-400" />
                    )}
                  </div>
                  <p className="text-gray-300 capitalize">
                    Status: <span className="font-semibold text-purple-300">{results.status}</span>
                  </p>
                </div>

                {/* Extracted Features */}
                {results.extracted_data && (
                  <div className="bg-slate-800 rounded-lg p-6 border border-purple-500/20">
                    <h3 className="text-lg font-semibold text-white mb-4">Code Features</h3>
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-gray-400">Languages</p>
                        <div className="flex flex-wrap gap-2 mt-1">
                          {results.extracted_data.languages?.map((lang) => (
                            <span key={lang} className="px-3 py-1 bg-purple-600 text-white text-sm rounded-full">
                              {lang}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Lines of Code</p>
                        <p className="text-white font-semibold">{results.extracted_data.code_length}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Functions Found</p>
                        <p className="text-white">{results.extracted_data.functions?.length || 0}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-400">Has Tests</p>
                        <p className="text-white">{results.extracted_data.has_tests ? '✓ Yes' : '✗ No'}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* ML Results */}
                {results.ml_result && (
                  <div className="bg-slate-800 rounded-lg p-6 border border-purple-500/20">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                      <Zap className="w-5 h-5 text-yellow-400" />
                      AI Analysis Results
                    </h3>
                    <div className="space-y-4">
                      <div className="grid grid-cols-2 gap-4">
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <p className="text-xs text-gray-400">Quality Score</p>
                          <p className="text-2xl font-bold text-purple-400">
                            {(results.ml_result.quality_score * 100).toFixed(0)}%
                          </p>
                        </div>
                        <div className="bg-slate-700 p-3 rounded-lg">
                          <p className="text-xs text-gray-400">Maintainability</p>
                          <p className="text-2xl font-bold text-green-400">
                            {(results.ml_result.maintainability * 100).toFixed(0)}%
                          </p>
                        </div>
                      </div>

                      {results.ml_result.issues && results.ml_result.issues.length > 0 && (
                        <div>
                          <p className="text-sm font-semibold text-red-300 mb-2">Issues Found</p>
                          <ul className="space-y-1">
                            {results.ml_result.issues.map((issue, i) => (
                              <li key={i} className="text-sm text-gray-300 flex gap-2">
                                <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
                                {issue}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {results.ml_result.suggestions && results.ml_result.suggestions.length > 0 && (
                        <div>
                          <p className="text-sm font-semibold text-blue-300 mb-2">Suggestions</p>
                          <ul className="space-y-1">
                            {results.ml_result.suggestions.map((suggestion, i) => (
                              <li key={i} className="text-sm text-gray-300 flex gap-2">
                                <CheckCircle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
                                {suggestion}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}