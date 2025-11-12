public class FibonacciNonRecursive 
{

    public static void main(String[] args) 
    {
        int n = 10; 
        int first = 0, second = 1, next;

        System.out.println("Fibonacci Series up to " + n + " terms:");

        for (int i = 0; i < n; i++) 
        {
            System.out.print(first + " ");
            next = first + second;
            first = second;
            second = next;
        }
    }
}
