function generateDeck() {
    const suits = ['♠', '♥', '♦', '♣'];
    const ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'];
    const deck = [];
  
    for (let suit of suits) {
        for (let rank of ranks) {
            deck.push({ rank, suit });
        }
    }
  
    return deck;
}
  
function shuffleDeck(deck) {
    for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
    }
}
  
function dealCards(deck, numCards) {
    return deck.slice(0, numCards);
}
  
function checkHand(hand) {
    const ranks = hand.map(card => card.rank);
    const counts = ranks.reduce((acc, rank) => {
        acc[rank] = (acc[rank] || 0) + 1;
        return acc;
    }, {});
  
    const duplicates = Object.values(counts).reduce((acc, count) => {
      if (count > 1) acc.push(count);
      return acc;
    }, []);
  
    // 스트레이트 플러쉬
    if (duplicates.includes(4)) return "Four of a Kind";
    if (duplicates.includes(3) && duplicates.includes(2)) return "Full House";
    // 플러쉬
    // 스트레이트
    if (duplicates.includes(3)) return "Three of a Kind";
    if (duplicates.length === 2) return "Two Pair";
    if (duplicates.includes(2)) return "One Pair";
    return "No Combination";
}
  
function main() {
    const deck = generateDeck();
    shuffleDeck(deck);
    const hand = dealCards(deck, 5);
  
    console.log("Your hand:", hand.map(card => `${card.suit}${card.rank}`).join(', '));
    console.log("Combination:", checkHand(hand));
}
  
main();
  