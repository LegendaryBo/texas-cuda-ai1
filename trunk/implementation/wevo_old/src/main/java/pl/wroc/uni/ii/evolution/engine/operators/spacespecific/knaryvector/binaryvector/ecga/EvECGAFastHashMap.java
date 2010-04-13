package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

/**
 * Simple and fast implementation of data struture similar to hash map which
 * counts ints given in put(i) method and returns numbers of the same values.
 * Implementation is fast and produces hardly any garbage. * hash function = x %
 * size; It's used in ECGA when counting same individuals.
 * 
 * @author Kacper Gorski
 */
public class EvECGAFastHashMap {

  // buffor int which stores key of the int being put (see public void put(int
  // i) )
  private int buff_key;

  // buffor int which iterates through the loops
  private int iter_buff;

  // stores values of int put in the hashmap
  private int[][] hashTable;

  // stores number of ints in adjecting table
  private int[][] hashTable_counts;

  // number of collisions-1 in adjecting table
  private int[] collisions;

  // size of the hashmap (it's always static)
  private int size;

  // starting size tables which store colliding ints
  private final int collision_size = 5;


  /**
   * Initializes hasmap
   * 
   * @param size of the hashmap hash function = x % size;
   */
  public EvECGAFastHashMap(int size) {
    this.size = size;
    hashTable = new int[size][collision_size];
    hashTable_counts = new int[size][collision_size];
    collisions = new int[size];
  }


  /**
   * puts 'i' to hashmap.
   * 
   * @param i int value
   */
  public void put(int i) {
    // hashing function
    buff_key = i % size;
    // when key has negative values
    if (buff_key < 0)
      buff_key -= buff_key;
    // searching all integer in hashmap which has the same hashing value
    for (iter_buff = 0; iter_buff < collisions[buff_key]; iter_buff++) {
      if (i == hashTable[buff_key][iter_buff]) {
        hashTable_counts[buff_key][iter_buff]++;
        return; // found, quit
      }
    }
    // not found

    // resize table when necessary
    if (hashTable[buff_key].length == collisions[buff_key]) {
      hashTable[buff_key] = double_size(hashTable[buff_key]);
      hashTable_counts[buff_key] = double_size(hashTable_counts[buff_key]);
    }
    // given int is the first one
    collisions[buff_key]++;
    hashTable[buff_key][iter_buff] = i;
    hashTable_counts[buff_key][iter_buff] = 1;
  }


  /**
   * Returns numbers of same int in hashmap. They are given in order of their
   * hash values.
   * 
   * @return
   */
  public int[] getCount() {
    int sum = 0;
    // summing numbers of differen values
    for (iter_buff = 0; iter_buff < size; iter_buff++)
      sum += collisions[iter_buff];
    int[] ret_table = new int[sum];

    int j = 0;
    int k = 0;
    // writing counts to the table
    for (iter_buff = 0; iter_buff < size; iter_buff++) {
      for (j = 0; j < collisions[iter_buff]; j++) {
        ret_table[k] = hashTable_counts[iter_buff][j];
        k++;
      }
    }
    return ret_table;
  }


  /**
   * Clears whole hashmap. Now there isn't anything here, but it still has the
   * same size
   */
  public void clear() {
    for (iter_buff = 0; iter_buff < size; iter_buff++) {
      collisions[iter_buff] = 0;
    }
  }


  // doubling size of the given table rewritting values to a new one
  private int[] double_size(int[] tab) {
    int tab_length = tab.length;
    int[] new_tab = new int[2 * tab_length];

    for (int i = 0; i < tab_length; i++) {
      new_tab[i] = tab[i];
    }
    return new_tab;
  }

}
