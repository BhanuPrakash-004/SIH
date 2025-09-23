import React, { useState } from 'react';

const Qa = ({ onSubmit, isLoading, response }) => {
    const [query, setQuery] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim()) {
            onSubmit(query);
            // ## FIX ##: Clear the input field after submitting the question
            setQuery(query); 
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="e.g., 'Summarize all safety reports from last month'"
                    className="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500"
                    disabled={isLoading}
                />
                <button 
                    type="submit" 
                    disabled={isLoading || !query.trim()} 
                    className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg shadow-md disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {isLoading ? 'Searching...' : 'Ask'}
                </button>
            </form>

            {/* ## CHANGE ##: Updated logic to render the response object */}
            {response && (
                <div className="mt-8">
                    <h3 className="text-2xl font-semibold text-gray-800 mb-3">💡 Answer</h3>
                    <div className="p-5 bg-blue-50 rounded-lg border border-blue-200">
                        <p className="text-gray-700 text-base whitespace-pre-wrap">{response.answer}</p>
                    </div>

                    {response.sources && response.sources.length > 0 && (
                        <div className="mt-6">
                             <h4 className="text-xl font-semibold text-gray-800 mb-3">📄 Sources</h4>
                             <div className="space-y-2">
                                {response.sources.map((source, index) => (
                                    <div key={index} className="p-3 bg-gray-100 rounded-lg border border-gray-200 flex items-center">
                                       <svg className="w-4 h-4 mr-2 text-gray-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"></path></svg>
                                       <p className="text-sm text-gray-600 font-mono">{source}</p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default Qa;