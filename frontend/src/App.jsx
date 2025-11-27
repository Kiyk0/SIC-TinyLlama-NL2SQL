import React, { useState } from 'react';
import { Search, Database, AlertCircle, CheckCircle, Loader, Code } from 'lucide-react';

export default function App() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const exampleQuestions = [
    "Show all students",
    "Find courses with rating above 4.5",
    "List all instructors and their specializations",
    "Show students enrolled in 'Machine Learning Fundamentals'",
    "What is the average price of courses?",
    "Find courses in the 'Data Science' category",
    "Show top 5 most popular courses",
    "List students who have completed a course"
  ];

  const handleSubmit = async () => {
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await fetch('http://localhost:5000/api/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question.trim() }),
      });

      const data = await res.json();

      if (data.success) {
        setResponse(data);
      } else {
        setError(data.error || 'An error occurred');
      }
    } catch (err) {
      setError('Failed to connect to the server. Make sure the backend is running on http://localhost:5000');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example) => {
    setQuestion(example);
    setError(null);
    setResponse(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      <div className="container mx-auto px-4 py-6">
        {/* Samsung Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-6">
            <img
              src="/samsung-logo.png"
              alt="Samsung"
              className="h-12 w-auto"
            />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2 tracking-tight">
            Smart Database Query
          </h1>
          <p className="text-blue-200 text-lg">Ask questions in natural language, powered by AI</p>
        </div>

        {/* Main Card - Samsung Style */}
        <div className="max-w-6xl mx-auto bg-white rounded-3xl shadow-2xl overflow-hidden mb-6">
          <div className="bg-gradient-to-r from-[#1428A0] to-[#0d6efd] p-6">
            <h2 className="text-white text-xl font-semibold">Query Assistant</h2>
            <p className="text-blue-100 text-sm mt-1">Intelligent database exploration</p>
          </div>

          <div className="p-6">
            {/* Query Input */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-gray-700 mb-3">
                What would you like to know?
              </label>
              <div className="flex gap-3">
                <input
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask anything about your database..."
                  className="flex-1 px-5 py-4 border-2 border-gray-200 rounded-2xl focus:ring-2 focus:ring-[#1428A0] focus:border-[#1428A0] outline-none transition-all text-gray-800 placeholder-gray-400"
                  disabled={loading}
                />
                <button
                  onClick={handleSubmit}
                  disabled={loading}
                  className="px-8 py-4 bg-gradient-to-r from-[#1428A0] to-[#0d6efd] text-white rounded-2xl hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 font-semibold"
                >
                  {loading ? (
                    <>
                      <Loader className="w-5 h-5 animate-spin" />
                      Processing
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Search
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Example Questions */}
            <div className="mb-6">
              <p className="text-sm font-semibold text-gray-700 mb-3">Quick Examples:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQuestions.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleExampleClick(example)}
                    className="px-4 py-2 text-sm bg-gradient-to-r from-blue-50 to-indigo-50 text-[#1428A0] rounded-full hover:from-blue-100 hover:to-indigo-100 transition-all border border-blue-200 font-medium"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="mb-6 p-5 bg-red-50 border-l-4 border-red-500 rounded-2xl flex items-start gap-3">
                <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="font-bold text-red-900 text-lg">Error</h3>
                  <p className="text-red-700 mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Success Response */}
            {response && (
              <div className="space-y-6">
                {/* Success Message */}
                <div className="p-5 bg-gradient-to-r from-green-50 to-emerald-50 border-l-4 border-green-500 rounded-2xl flex items-center gap-3">
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <div>
                    <h3 className="font-bold text-green-900 text-lg">Success</h3>
                    <p className="text-green-700">Found {response.row_count} result{response.row_count !== 1 ? 's' : ''}</p>
                  </div>
                </div>

                {/* Generated SQL */}
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Code className="w-5 h-5 text-[#1428A0]" />
                    <h3 className="font-bold text-gray-800 text-lg">Generated SQL Query</h3>
                  </div>
                  <div className="bg-gradient-to-br from-gray-900 to-slate-800 text-green-400 p-5 rounded-2xl overflow-x-auto border border-gray-700">
                    <code className="text-sm font-mono">{response.generated_sql}</code>
                  </div>
                </div>

                {/* Results Table */}
                {response.data && response.data.length > 0 && (
                  <div>
                    <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2 text-lg">
                      <Database className="w-6 h-6 text-[#1428A0]" />
                      Results ({response.row_count} rows)
                    </h3>
                    <div className="overflow-x-auto border-2 border-gray-200 rounded-2xl">
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gradient-to-r from-[#1428A0] to-[#0d6efd]">
                          <tr>
                            {response.columns.map((col, idx) => (
                              <th
                                key={idx}
                                className="px-6 py-4 text-left text-xs font-bold text-white uppercase tracking-wider"
                              >
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-100">
                          {response.data.map((row, rowIdx) => (
                            <tr key={rowIdx} className="hover:bg-blue-50 transition-colors">
                              {response.columns.map((col, colIdx) => (
                                <td key={colIdx} className="px-6 py-4 whitespace-nowrap text-sm text-gray-800 font-medium">
                                  {row[col] !== null && row[col] !== undefined ? String(row[col]) : '-'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* No Results */}
                {response.data && response.data.length === 0 && (
                  <div className="text-center py-12 text-gray-500">
                    <Database className="w-16 h-16 mx-auto mb-3 opacity-30" />
                    <p className="text-lg">No results found for your query</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Footer Info - Samsung Style */}
        <div className="max-w-6xl mx-auto bg-white/10 backdrop-blur-lg border border-white/20 rounded-3xl p-6">
          <h4 className="font-bold text-white mb-3 text-lg">How it works</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-100">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1428A0] flex items-center justify-center flex-shrink-0 font-bold text-white">1</div>
              <p>Type your question in natural language</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1428A0] flex items-center justify-center flex-shrink-0 font-bold text-white">2</div>
              <p>AI converts it to a SQL query</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1428A0] flex items-center justify-center flex-shrink-0 font-bold text-white">3</div>
              <p>Query executes safely in read-only mode</p>
            </div>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-[#1428A0] flex items-center justify-center flex-shrink-0 font-bold text-white">4</div>
              <p>Results displayed in a clean table</p>
            </div>
          </div>
        </div>

        {/* Samsung Footer */}
        <div className="text-center mt-8">
          <p className="text-blue-200 text-sm">Powered by Samsung Innovation</p>
        </div>
      </div>
    </div>
  );
}