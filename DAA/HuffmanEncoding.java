import java.util.Comparator;
import java.util.PriorityQueue;

class HuffmanNode 
{
    int data;
    char c;
    HuffmanNode left, right;
}

class MyComparator implements Comparator<HuffmanNode> 
{
    @Override
    public int compare(HuffmanNode x, HuffmanNode y) 
    {
        return x.data - y.data;
    }
}

public class HuffmanEncoding 
{
    public static void printCode(HuffmanNode root, String s) 
    {
        if (root.left == null && root.right == null && Character.isLetter(root.c)) 
        {
            System.out.println(root.c + " : " + s);
            return;
        }
        printCode(root.left, s + "0");
        printCode(root.right, s + "1");
    }

    public static void main(String[] args) 
    {
        char[] charArray = {'a', 'b', 'c', 'd', 'e', 'f'};
        int[] charFreq = {5, 9, 12, 13, 16, 45};

        PriorityQueue<HuffmanNode> q = new PriorityQueue<>(charArray.length, new MyComparator());

        for (int i = 0; i < charArray.length; i++) 
        {
            HuffmanNode node = new HuffmanNode();
            node.c = charArray[i];
            node.data = charFreq[i];
            node.left = null;
            node.right = null;
            q.add(node);
        }

        HuffmanNode root = null;
        while (q.size() > 1) 
        {
            HuffmanNode x = q.poll();
            HuffmanNode y = q.poll();

            HuffmanNode f = new HuffmanNode();
            f.data = x.data + y.data;
            f.c = '-'; 
            f.left = x;
            f.right = y;
            root = f;
            q.add(f);
        }

        System.out.println("Character Huffman Codes:");
        printCode(root, "");
    }
}