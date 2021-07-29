import React, { useEffect, useRef, useState } from 'react';
import { Board } from './board';

interface Key {
    seed: number;
    index: number;
}

interface GameLogicPropsState {
    board: string;
    score: number;
}

interface GameLogicProps {
    seed: number;
    numGames: number;
}

const GameLogic: React.FC<GameLogicProps> = (props) => {
    const [state, setState] = useState<GameLogicPropsState>({'board': '0000000000000000', 'score': 0});
    const key = useRef<Key>({'seed': 0, 'index': 0})

    useEffect(() => {
        const initState = async() => {
            [key.current.seed, key.current.index] = [props.seed, 0]
            const randQ = `http://localhost:8000/random?seed=${key.current.seed}&index=${key.current.index}&n=${2}`;
            const [newKey, subKey] = await fetch(randQ).then(res => res.json());
            const initQ = `http://localhost:8000/game/init?seed=${subKey.seed}&index=${subKey.index}`;
            const newBoard = await fetch(initQ).then(res => res.json());
            [key.current.seed, key.current.index] = [newKey.seed, newKey.index];
            setState({'board': newBoard, 'score': 0});
        }

        initState();
    }, [props.seed, props.numGames])

    useEffect(() => {
        const execState = async(action: number) => {
            const randQ = `http://localhost:8000/random?seed=${key.current.seed}&index=${key.current.index}&n=${2}`;
            const [newKey, subKey] = await fetch(randQ).then(res => res.json());
            const execQ = `http://localhost:8000/game/exec?seed=${subKey.seed}&index=${subKey.index}&state=${state.board}&action=${action}`;
            const turn = await fetch(execQ).then(res => res.json());
            [key.current.seed, key.current.index] = [newKey.seed, newKey.index];
            setState({'board': turn.nextState, 'score': state.score + turn.reward});
        }

        const handleKeyPress = (event: KeyboardEvent) => {
            switch(event.key) {
                case 'ArrowLeft':
                    execState(0);
                    break;
                case 'ArrowUp':
                    execState(1);
                    break;
                case 'ArrowRight':
                    execState(2);
                    break;
                case 'ArrowDown':
                    execState(3);
                    break;
            }
        }

        document.addEventListener('keydown', handleKeyPress)

        return () => {document.removeEventListener('keydown', handleKeyPress)}
    })

    return <Board board={ state.board } score={ state.score }/>
}

interface SeedInputProps {
    seed: number;
    onChange: React.Dispatch<React.SetStateAction<number>>;
    onClick: () => void;
}

const SeedInput: React.FC<SeedInputProps> = (props) => {
    const [seed, setSeed] = useState(props.seed);
    const randomSeed = () => {
        setSeed(Math.floor(Math.random() * 1e9));
    }
    const newGame = () => {
        props.onChange(seed);
        props.onClick();
    }

    return <>
        <div className='seed-input'>
            <input type='number' value={ seed } onChange={ e => setSeed(parseInt(e.target.value)) }/>
            <button onClick={ randomSeed }>Random Seed</button>
            <button onClick={ newGame }>New Game</button>
        </div>
    </>
}

export const Game: React.FC = () => {
    const [seed, setSeed] = useState(0);
    const [numGames, setNumGames] = useState(0);
    const incrementGames = () => setNumGames(numGames + 1);

    return <>
        <div className='game'>
            <SeedInput seed = { seed } onChange = { setSeed } onClick = { incrementGames }/>
            <GameLogic seed = { seed } numGames = { numGames }/>
        </div>
    </>
}