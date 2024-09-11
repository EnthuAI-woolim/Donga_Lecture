// 240327_week4 - 개인실습 난이도2까지만 함
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

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
    
    const check =  checkHand(hand)
    console.log("Your hand:", hand.map(card => `${card.suit}${card.rank}`).join(', '));
    console.log("Combination:", check);
    console.log('');

    combinationNum[check]++;
}

let combinationNum = {
    "Four of a Kind": 0, "Full House": 0, "Three of a Kind": 0, "Two Pair": 0, "One Pair": 0, "No Combination": 0}

rl.question('게임을 몇번 하시겠습니까? : ', (input) => {
    const num = parseInt(input);

    console.log('');
    for (let i = 0; i < num; i++){
        main();
    }
    
    for (const [combination, count] of Object.entries(combinationNum)) {
        console.log(combination + ' : ' + count);
    }

    rl.close();
  });
  
