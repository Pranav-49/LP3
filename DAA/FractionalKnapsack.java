import java.util.*;

class Item 
{
    int value, weight;
    Item(int value, int weight) 
    {
        this.value = value;
        this.weight = weight;
    }
}

public class FractionalKnapsack 
{
    public static double getMaxValue(Item[] items, int capacity) 
    {
        Arrays.sort(items, new Comparator<Item>() 
        {
            public int compare(Item a, Item b) 
            {
                double r1 = (double) a.value / a.weight;
                double r2 = (double) b.value / b.weight;
                return Double.compare(r2, r1);
            }
        });

        double totalValue = 0.0;

        for (Item i : items) 
        {
            if (capacity - i.weight >= 0) 
            {
                capacity -= i.weight;
                totalValue += i.value;
            } 
            else {
                double fraction = (double) capacity / i.weight;
                totalValue += i.value * fraction;
                capacity = 0;
                break; 
            }
        }

        return totalValue;
    }

    public static void main(String[] args) 
    {
        Item[] items = {
            new Item(60, 10),
            new Item(100, 20),
            new Item(120, 30)
        };
        int capacity = 50;
        double maxValue = getMaxValue(items, capacity);
        System.out.println("Maximum value in Knapsack = " + maxValue);
    }
}

//  Total Time Complexity : O(n log n)
//  Space Complexity      : O(1)
