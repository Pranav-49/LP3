public class NQueens 
{

    static final int N = 4; 

    static void printSolution(int board[][]) 
    {
        for (int i = 0; i < N; i++) 
        {
            for (int j = 0; j < N; j++)
                System.out.print((board[i][j] == 1 ? "Q " : ". "));
            System.out.println();
        }
    }

    static boolean isSafe(int board[][], int row, int col) 
    {
        int i, j;

        for (i = 0; i < col; i++)
            if (board[row][i] == 1)
                return false;

        for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
        {
            if (board[i][j] == 1)
                return false;
        }

        for (i = row, j = col; j >= 0 && i < N; i++, j--)
        {
            if (board[i][j] == 1)
                return false;
        }
        return true;
    }

    static boolean solveNQUtil(int board[][], int col) 
    {
        if (col >= N)
            return true;

        for (int i = 0; i < N; i++) 
        {
            if (isSafe(board, i, col)) 
            {
                board[i][col] = 1; 

                if (solveNQUtil(board, col + 1))
                    return true;

                board[i][col] = 0; 
            }
        }
        return false; 
    }

    static boolean solveNQ() 
    {
        int board[][] = new int[N][N];

        board[0][0] = 1;

        if (solveNQUtil(board, 1) == false) 
        {
            System.out.println("Solution does not exist");
            return false;
        }

        printSolution(board);
        return true;
    }

    public static void main(String args[]) 
    {
        solveNQ();
    }
}
