import Image from "next/image";
import iconError from "../../public/Icon3.svg"

export default function Banner() {
    return (
        <div className="h-14 bg-warningCustom flex justify-start w-full mx-auto px-2.5 gap-3" style={{
            background: 'linear-gradient(90deg, rgba(220, 104, 3, 0.09) 0%, rgba(220, 104, 3, 0.09) 100%)',
            alignItems: 'center'
        }}
        >
            <Image
                src={iconError}
                alt="error icon"
                width={20}
                height={20}
            />
            <p className="text-warning trial-session">This is a FREE trial session 2 of 5</p>
        </div>
    );
}
