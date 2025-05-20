import React, { useState, useEffect } from 'react';
import { ChevronRight, ChevronDown, File, Folder } from 'lucide-react';

const FileHierarchyItem = ({ 
  item, 
  depth = 0, 
  onViewFile,
  isRoot = false
}) => {
  const [isExpanded, setIsExpanded] = useState(isRoot);

  useEffect(() => {
    if (isRoot) {
      setIsExpanded(true);
    }
  }, [isRoot]);

  const toggleExpand = () => {
    if (item.type === 'directory') {
      setIsExpanded(!isExpanded);
    } else if (item.type === 'file') {
      onViewFile(item);
    }
  };

  const renderIcon = () => {
    if (item.type === 'directory') {
      return isExpanded ? (
        <ChevronDown className="w-6 h-6 text-teal-600" />
      ) : (
        <ChevronRight className="w-6 h-6 text-teal-600" />
      );
    }
    return <File className="w-6 h-6 text-teal-500" />;
  };

  return (
    <div>
      <div 
        className={`flex items-center cursor-pointer p-4 rounded-lg transition-all duration-300 hover:bg-teal-100 hover:shadow-lg ${
          item.type === 'directory' ? 'font-semibold text-teal-800' : 'text-teal-700'
        }`}
        style={{ paddingLeft: `${depth * 32}px` }}
        onClick={toggleExpand}
      >
        <span className="mr-4">{renderIcon()}</span>
        <span className="flex-grow text-lg truncate">{item.name}</span>
        <span className={`text-sm px-3 py-1 rounded-full ${
          item.type === 'directory' ? 'bg-teal-200 text-teal-800' : 'bg-teal-100 text-teal-700'
        }`}>
          {item.type === 'directory' ? 'Folder' : 'File'}
        </span>
      </div>
      
      {item.type === 'directory' && isExpanded && item.children && (
        <div className="mt-2 ml-4">
          {item.children.length > 0 ? (
            item.children.map((child, index) => (
              <FileHierarchyItem 
                key={index} 
                item={child} 
                depth={depth + 1}
                onViewFile={onViewFile}
                isRoot={false}
              />
            ))
          ) : (
            <div className="text-teal-600 text-sm pl-8 py-2">
              No items in this folder
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default function FileHierarchy({ 
  fileHierarchy, 
  onViewFile 
}) {
  return (
    <div className="w-full bg-white rounded-xl shadow-2xl p-8 border border-teal-200/30 overflow-auto">
      {fileHierarchy ? (
        <FileHierarchyItem 
          item={fileHierarchy} 
          onViewFile={onViewFile}
          isRoot={true}
        />
      ) : (
        <div className="text-teal-600 text-center py-12 text-lg">
          No project structure available
        </div>
      )}
    </div>
  );
}