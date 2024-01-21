import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "What is the total sum of the budget for all districts in 2024, including economic social assistance?",
    "Hva er forskjellen i budsjettet for økonomisk sosialhjelp mellom Alna og Nordstrand i 2024?",
    "Hva er prosentandelen av barnehagelærere i grunnbemanningen i kommunale barnehager i 2022, og hva er målet for 2024?"
];

const GPT4V_EXAMPLES: string[] = [
    "What is the total sum of the budget for all districts in 2024, including economic social assistance?",
    "Hva er forskjellen i budsjettet for økonomisk sosialhjelp mellom Alna og Nordstrand i 2024?",
    "Hva er prosentandelen av barnehagelærere i grunnbemanningen i kommunale barnehager i 2022, og hva er målet for 2024?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked, useGPT4V }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {(useGPT4V ? GPT4V_EXAMPLES : DEFAULT_EXAMPLES).map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
