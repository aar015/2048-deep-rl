import React from 'react';

interface Props {
    board: string;
    score: number;
}

export const Board: React.FC<Props> = (props) => {
    return <>
        <div className='score'> { props.score } </div>
        <div className='board-box'>
            <div className='board'>
                <div className={ `tile${ props.board[0]  }` }/>
                <div className={ `tile${ props.board[1]  }` }/>
                <div className={ `tile${ props.board[2]  }` }/>
                <div className={ `tile${ props.board[3]  }` }/>
                <div className={ `tile${ props.board[7]  }` }/>
                <div className={ `tile${ props.board[6]  }` }/>
                <div className={ `tile${ props.board[5]  }` }/>
                <div className={ `tile${ props.board[4]  }` }/>
                <div className={ `tile${ props.board[8]  }` }/>
                <div className={ `tile${ props.board[9]  }` }/>
                <div className={ `tile${ props.board[10] }` }/>
                <div className={ `tile${ props.board[11] }` }/>
                <div className={ `tile${ props.board[15] }` }/>
                <div className={ `tile${ props.board[14] }` }/>
                <div className={ `tile${ props.board[13] }` }/>
                <div className={ `tile${ props.board[12] }` }/>
            </div>
        </div>
    </>
}
