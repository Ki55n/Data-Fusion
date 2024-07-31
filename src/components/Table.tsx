import React from 'react';

// Utility function to generate a random color
const getRandomColor = () => {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
        color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
};

interface FilterProps {
    name: string;
    icon?: React.ReactNode;
}
export interface TableProps {
  title: string;
  columns: TableColumn[];
  data: TableRow[];
  rowUrl?: string;
  filters?: FilterProps[];
}
export interface TableColumn {
  key: string;
  title: string;
  width: string;
}
export interface TableRow {
  [key: string]: string | React.ReactNode;
}

export default function Table({columns, data}: TableProps) {
    return (
        <div className="w-full">
            <table className="w-full px-4 border-primary border-1 rounded mt-4">
                <thead className={'border-primary border-1'}>
                    <tr className="bg-darkViolet h-8">
                        {columns.map((column, index) => (
                            <th key={index} className={`${column.width} text-left px-8`}>{column.title}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row, index) => (
                        <tr key={index} className="h-9">
                            {row.rowUrl ? (
                                <a href={row.rowUrl} className="w-full contents">
                                    {columns.map((column, colIndex) => (
                                        <td key={colIndex} className={`px-8 py-5 ${column.width}`}>
                                            {column.key === 'name' ? (
                                                <div className="flex items-center gap-2">
                                                    <div
                                                        className="w-4 h-4"
                                                        style={{ backgroundColor: getRandomColor() }}
                                                    />
                                                    {row[column.key]}
                                                </div>
                                            ) : (
                                                row[column.key]
                                            )}
                                        </td>
                                    ))}
                                </a>
                            ) : (
                                <>
                                    {columns.map((column, colIndex) => (
                                        <td key={colIndex} className={`px-8 py-5 ${column.width}`}>
                                            {column.key === 'name' ? (
                                                <div className="flex items-center gap-2">
                                                    <div
                                                        className="w-4 rounded h-4"
                                                        style={{ backgroundColor: getRandomColor() }}
                                                    />
                                                    {row[column.key]}
                                                </div>
                                            ) : (
                                                row[column.key]
                                            )}
                                        </td>
                                    ))}
                                </>
                            )}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}
