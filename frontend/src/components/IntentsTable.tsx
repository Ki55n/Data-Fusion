import Image from "next/image";

const data = [
    {
        intentName:  "Post-Sale Stakeholder Kickoff & Onboarding Meeting",
        clientName: "JP Morgan",
        date: "06/26/2021"
    }
]

export default function IntentsTable() {
    return (
        <div>
            <div className='w-full flex justify-between items-center gap-4 py-3 px-7'>
                <p>Created Set Intents</p>
                <div className="flex justify-around">
                    <div className='relative'>
                        <Image
                            src={'/search.svg'}
                            alt='search'
                            width={16}
                            height={16}
                            className='absolute left-3 top-1/2 transform -translate-y-1/2'
                        />
                        <input type="search" placeholder='Type here...' className='search-input pl-8  pr-4 mr-3.5'/>
                    </div>
                    <button>
                        <Image
                            src={'/Settings.png'}
                            alt='search'
                            width={40}
                            height={40}
                        />
                    </button>
                </div>
            </div>
            <div className="w-full">
                <table className="w-full table-fixed px-4">
                    <thead>
                        <tr className=" bg-darkViolet h-8">
                            <th className="w-3/6 text-left px-8">Intent name</th>
                            <th className="w-2/6 text-left px-8">Client name</th>
                            <th className="w-1/6 text-left px-8">Date</th>
                            <th className="w-1/6 text-left "></th>
                            <th className="w-1/6"></th>
                        </tr>
                    </thead>
                    <tbody>
                        {data.map((row, index) => (
                            <tr key={index} className="h-9">
                                <td className="w-3/6 px-8 py-5">{row.intentName}</td>
                                <td className="w-1/6 px-8">{row.clientName}</td>
                                <td className="w-1/6 px-8">{row.date}</td>
                                <td className="w-1/6">
                                    <div className="flex justify-around">
                                        <Image
                                            src={'/edit.png'}
                                            alt='edit'
                                            width={24}
                                            height={24}
                                        />
                                        <Image
                                            src={'/copy.png'}
                                            alt='copy'
                                            width={24}
                                            height={24}
                                        />
                                        <Image
                                            src={'/Cross Circle.png'}
                                            alt='cross circle'
                                            width={24}
                                            height={24}
                                        />
                                    </div>
                                </td>
                                <td className="w-1/6 px-8">
                                    <div>
                                        <Image
                                            src={'/Three dots.png'}
                                            alt='three dots'
                                            width={24}
                                            height={24}
                                        />
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="py-3 px-5">
                <button className="btn-intents">show all created intents</button>
            </div>
        </div>
    )
}
