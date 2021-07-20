import React, { useEffect, useRef, useState } from 'react';
import { Board } from './board';

interface Key {
    seed: number,
    index: number,
}

interface GameLogicPropsState {
    board: string,
    score: number,
}

interface GameLogicProps {
    seed: number 
}

const GameLogic: React.FC<GameLogicProps> = (props) => {
    const [state, setState] = useState<GameLogicPropsState>({'board': '0000000000000000', 'score': 0});
    const key = useRef<Key>({'seed': props.seed, 'index': 0})

    useEffect(() => {
        const initState = async() => {
            const randQ = `http://localhost:8000/random?seed=${key.current.seed}&index=${key.current.index}&n=${2}`;
            const [newKey, subKey] = await fetch(randQ).then(res => res.json());
            const initQ = `http://localhost:8000/game/init?seed=${subKey.seed}&index=${subKey.index}`;
            const newBoard = await fetch(initQ).then(res => res.json());
            [key.current.seed, key.current.index] = [newKey.seed, newKey.index];
            setState({'board': newBoard, 'score': 0});
        }

        initState();
    }, [])

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
}

const SeedInput: React.FC<SeedInputProps> = (props) => {
    return <>
        <div className='seed-input'>
            <label htmlFor='seed'>Seed</label>
            <input type='number' id='seed'/>
            <button>Random Seed</button>
            <button>New Game</button>
        </div>
    </>
}

export const Game: React.FC = () => {
    const [seed, setSeed] = useState(0);
    return <>
        <div className='game'>
            <SeedInput seed = { seed } onChange = { setSeed }/>
            <GameLogic seed = { seed }/>
        </div>
    </>
}