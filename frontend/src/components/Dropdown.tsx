import React from "react";

interface DropdownProps {
    options: {label: string, value: string}[];
    // eslint-disable-next-line no-unused-vars
    onChange: (event: React.ChangeEvent<HTMLSelectElement>) => void;
    value: string;
}
export default function Dropdown({options, onChange, value}: DropdownProps) {
    return (
        <select className="p-2 h-12 border rounded flex items-center w-full" onChange={onChange} value={value}>
            {options.map((option, index) => (
                <option key={index} value={option.value} className="h-12 p-4">
                    {option.label}
                </option>
            ))}
        </select>
    )
}
