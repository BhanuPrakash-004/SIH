//frontend-react/src/components/Summary.jsx
import React from 'react';

const Summary = ({ summary, takeaways }) => {
    if (!summary && !takeaways) {
        return null;
    }

    return (
        <div className="mt-6 space-y-6">
            {summary && (
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">📄 Executive Summary</h3>
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <p className="text-gray-600 whitespace-pre-wrap">{summary}</p>
                    </div>
                </div>
            )}
            {takeaways && (
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">📌 Key Takeaways</h3>
                    <div className="p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <div className="text-gray-600 whitespace-pre-wrap" dangerouslySetInnerHTML={{ __html: takeaways.replace(/\n/g, '<br />') }} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default Summary;
