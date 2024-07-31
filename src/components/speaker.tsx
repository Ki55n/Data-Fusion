'use client'
export default function Speaker({text, child}: {text: string, child: React.ReactNode}) {
    const utterance = new SpeechSynthesisUtterance("Welcome to this tutorial!");
    const voices = speechSynthesis.getVoices();
    utterance.voice = voices[0];
    return (
        <button
            onClick={() => {
                    const synth = window.speechSynthesis;
                    const utterThis = new SpeechSynthesisUtterance(text);
                    synth.speak(utterThis);
            }}
        >
            {child}
        </button>
    )
}

