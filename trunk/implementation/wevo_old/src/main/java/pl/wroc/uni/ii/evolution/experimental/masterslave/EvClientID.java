package pl.wroc.uni.ii.evolution.experimental.masterslave;

/**
 * Contains client id.
 * 
 * @author Karol 'Asgaroth' Stosiek (karol.stosiek@gmail.com)
 * @author Mateusz 'm4linka' Malinowski (m4linka@gmail.com)
 */
public final class EvClientID {
  
  /**
   * Id number.
   */
  private int id;
  
  /**
   * Constructor.
   * 
   * @param client_id - client id to set
   */
  public EvClientID(final int client_id) {
    this.id = client_id;
  }

  /**
   * Gets client id.
   * 
   * @return client id
   */
  public int getId() {
    return this.id;
  }

  /**
   * Sets client id.
   * 
   * @param client_id - client id to set
   */
  public void setId(int client_id) {
    this.id = client_id;
  }
  
  /**
   * Returns client id converted to string.
   * 
   * @return client id converted to string
   */
  @Override
  public String toString() {
    return Integer.toString(this.id);
  }
}
