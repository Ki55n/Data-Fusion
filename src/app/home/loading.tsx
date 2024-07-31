import {ImSpinner9} from "react-icons/im";

export default function Loading() {
    return (
        <div className="w-full h-full flex justify-center items-center">
        <p className="text-2xl text-white">Loading...</p>
        <ImSpinner9 className="animate-spin text-primary text-2xl" />
        </div>
    );
}
