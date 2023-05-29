import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        //text: "What is included in my Northwind Health Plus plan that is not in standard?",
        //value: "What is included in my Northwind Health Plus plan that is not in standard?"
        text: "Hur återställer jag mitt lösenord för Canvas?",
        value: "Hur återställer jag mitt lösenord för Canvas?"
    },
    { 
        //text: "What happens in a performance review?", 
        //value: "What happens in a performance review?" 
        text: "Hur ansluter jag mig till eduroam?", 
        value: "Hur ansluter jag mig till eduroam?" 
    },
    { 
        //text: "What does a Product Manager do?", 
        //value: "What does a Product Manager do?"
        text: "Skriv en informativ Rap Battle om Högskolan Väst", 
        value: "Skriv en informativ Rap Battle om Högskolan Väst" 
    }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
