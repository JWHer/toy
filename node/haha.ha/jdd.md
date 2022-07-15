# JDD (Ju-Dung-A-Li Driven Development/주둥아리 주도 개발)
![image](https://user-images.githubusercontent.com/10369528/130068355-049861db-ce6e-45cd-a74c-a262ee4fb01a.png)

JDD 는 코드 작성보다 변명거리를 미리 생각하여 수많은 버그를 양성할 수 있는 개발 방법론이다.

JDD 는 아래와 같은 중요 가치를 따른다.

```
우리는

- 남이 쓰는 기술보다는 내 것을

- Clean 한 코드보다는 Tricky 한 것을

- 버그 Fix 보다는 변명을

- 귀찮음보다는 편함을

가치있게 여긴다. 이 말은, 왼쪽에 있는 것들은 귀찮고 오른쪽에 있는 것들은 편하기 때문에 더 높은 가치를 둔다는 것이다.
```

JDD 를 실천하기 위해서는 쉬지 않고 험난한 길이지만 꾸준하게 노력해야한다.
중요한 것은 관련된 힙한 용어를 들먹이면서 실제 문제에 대한 답변을 회피하는것이다. 꾸준한 연습이 필요하다.


# JDD 를 위한 지침서

## JDD 를 위한 소양

JDD 의 기초가 되는 소양을 항상 명심하자.

### 참된 리더
참된 리더는 부하직원을 괴롭히지 않는다.

```
- 누군가 PR 을 했다는 사실 자체가 보기 좋은 일이다. LGTM(Looks Good To Me).

- 누군가 PR 을 날리면 자동으로 LGTM 를 날리는 CI/CD 를 활용해라
```

### 방어적 프로그래밍
방어적 프로그래밍은 개발자의 기본 소양이다.

```
- 방어적 프로그래밍을 해라 코드로 방어하지 말고 너에게 들어오는 일감을 방어해라

- 작업양이 많은가? 그냥 특정 테스트 케이스에서만 돌아가게 짜고 나머지는 "고도화" 작업에서 한다고 해라.
"고도화" 작업 하기전에 이직하자.
```

## 성과 위주 회피

일 잘하는 척 편안한 생활을 위해 성과 위주로 행동해야한다. 짤리기는 싫으니까 !

### 멀티 쓰레드
동시성은 어렵고, 성과가 눈에 띄지 않는다.
```
- 멀티 쓰레드 환경에서 가변변수 사용으로 인한 오류는 가끔 발생한다.
가끔 발생한다는 거에 비하여 사용은 너무 편하다. 누가 뭐라고 하면 그럴 일 없다고 하면 그만이다.

- 어쩔 수 없이 멀티 쓰레드 업무가 들어오면 언어를 욕하면서 이 작업이 왜 어려운지 설명해라,
아 이 언어는 Actor 모델이 빈약해서요, CSP(Communicating Sequential Processes) 구현체가 없어서요, STM(Software Transactional Memory) 이 지원 안 돼서요.
욕을 할수록 작업 기한이 늘어난다.

- 비동기 흐름 내에서 동기함수를 몰래 써도 된다. 여기에 병목 찾는 작업 + 비동기로 개선 하는 작업하는 기한까지 나중에 받을 수 있다.
```

### 문서화
우리의 본분을 잊지 말자.

```
- 주석은 절대 작성하지 않는다. 누가 뭐라고 하면 클린코드를 들먹이자. 물론 그렇다고해서 코드가 리더블하지는 않다.

- 우리는 개발자다, 어떠한 이유에서든지 문서 작성은 금물. 누가 뭐라고 하면 "코드가 곧 문서".
```

### 협업
단어부터 현기증이 온다.

```
- 여럿이서 동시에 같은 코드를 작성해야한다면 어떻게든 나눠서 따로 작업한 뒤 합치자고하자. 상대방도 하나를 두고 부대끼며 작업할 생각따위 없다.

- 페어코딩은 의외로 괜찮다. 내가 코드를 짤 때에는 머리를 좀 비워도 되고, 다른 사람이 코드를 짤 때에는 오타 정도만 잡아주면 충분하다. 최대 장점은 작성된 코드에 대한 책임이 줄어든다는 점이다.

- 무슨 작업인지 별로 알리고 싶지않다면 커밋 메시지는 `chore: trivial`이라고 하자. 변태가 아니고서야 이런 커밋을 읽어보는 사람은 없다.

- PR 을 머지할 때에는 Squash merge 를 최대한 활용해서 중간중간에 들어간 뻘짓들을 숨긴다. 누가 뭐라고하면 '커밋 로그를 간략하게 유지하기 위해서'라고 하자.

- 커밋은 나눌 때는 작업이나 기능 단위가 아니라 내가 한 번에 고친만큼을 기준으로 한다. 어차피 머지되면 다 Squash 돼서 잘 안보인다.

- PR Description 은 작업 관리 툴의 task 링크만 티켓으로 넣어준다. 물론 링크를 타고가도 설명 같은건 없다.
```

## 뭐든 프로그래밍

세상은 0과 1로 이루어졌다고 할 수 있다. 그러니까 뭘 하든지 내가 하는 것은 프로그래밍인 것이다.

### 모던 프로그래밍
모던은 현재이다. 즉 지금 내가쓰는 기술이 곧 모던한 기술이다.

```
- 예외 처리를 하지 않는다. 누가 뭐라고 하면 Let it crash 전략이라고 한다.

- 언어의 Feature 를 최대한 사용하지 않는다. 누가 뭐라고 하면 유지보수를 위해 누구나 알 수 있게 짜둔 거라고 한다.

- 언어의 Feature 를 최대한 활용하자. 누가 뭐라고 하면 모던 프로그래밍이라고 하면 된다.

- 의존성이 복잡하게 연결되어 있으면, 패턴이라고 한다.

- 체이닝(Chaining)이 하나라도 있으면, Fluent API 스타일

- 체이닝이 하나도 없으면, 디미터 법칙

- 함수들만 호출하는 함수가 있으면, 미니 언어로 DSL 을 구축했다고 해라

- 데이터를 넘겨주는 함수가 있으면, Data-Driven Programming 이라고 해라

- 함수 파라미터가 너무 많아졌다? 누가 뭐라고 하면 보다 "순수"하게 짠 거라고 해라

- 람다 함수로만 구성한다. 누가 뭐라고 하면 고차함수 함수형 프로그래밍이라고 한다.

- 패턴을 전혀 쓰지 않는다. 누가 뭐라고 하면 단순성의 원칙이라고 한다.

- Map-Reduce-Filter 같은 고차함수는 일절 사용하지 않는다. 누가 뭐라고 하면 이 역시 단순성의 원칙이라고 한다. Keep it simple, Stupid!

- 어노테이션과 같은 메타데이터를 수십개는 달자. 누가 뭐라고 하면 메타 프로그래밍 기법이라고 한다.

- OOP(Object-Oriented Programming)로 개발하는 사람들 사이에서 FP 패턴(Functional Programming)을 적극 활용하자.
라이브러리까지 쓰면 더 좋다. Maybe 클래스를 만드는건 기본 중의 기본. 누가 뭐라고 하면 "이게 요즘 스타일이예요"

- FP 로 개발하는 사람들 사이에서 OOP 패턴과 가변변수를 적극 활용하자. 누가 뭐라고 하면, "이렇게 하는게 더 편한데요?"

- 콜백 헬을 보고 뭐냐고 물어보면 CPS(Continous Passing Style) 라고 답해줘라

- 모든 if 문은 삼항 연산자로 써라, 누가 뭐라고 하면, "아 if 는 문이고 삼항은 식이니까요"

- 리스트 컴프리헨션, 옵셔널 체이닝, 리엑티브 등을 쓰고 자랑스럽게 말해라 "모나딕하게 해결했다고".
그게 뭐냐고 물어보면 "모나드의 저주로 인하여 설명하기 힘들다" 까지 말하는 거까지 해야한다.

- 두 개 이상 실행되는 서비스가 있으면 MSA(Micro-Service Architecture) 다.
테스트를 돌려주는 스크립트만 따로 있어도 MSA 다.
거기에 언어가 다르면 폴리글랏(polyglot)까지 했다고 말할 수 있다.
```

### 설계
어떻게 설계해도 돌아간다.
```
- 클래스, 인터페이스, 이넘, 맴버등 구조 설계에 대해서 뭐라고 하면, ADT 에서 합타입이 어쩌고 곱타입이 어쩌고

- 클래스에 기능이 너무 적으면, 아 그건 레코드로 쓰려고 했습니다.

- 상속이 합성보다 편하다. 누가 뭐라고 하면 OOP 에서는 당연히 상속을 써야하는게 맞다고 하면된다.

- null 은 10억불의 가치가 있다. 뭐든 애매하면 null 을 리턴해라

- Object 나 Any 는 폴리몰피즘의 극의이다. 뭐든 애매하면 Object 타입을 리턴해라,
Object 타입을 리턴하는 함수에서 null 을 리턴하도록 하는게 베스트
```

### 테스트
솔직히 테스트코드 짜는 건 재미 없다
```
- 테스트코드를 전혀 작성하지 않는다. 누가 뭐라고 하면 repl 로 테스트가 끝났다고 해라.

- 테스트가 어떨때는 성공하고 어떨때는 실패하면, "속성 기반 테스트" 를 작성해서 그렇다고 해라
```

### 성능최적화
CPU 성능은 내 실력과 달리 날로 발전한다.
```
- DTO/VO 변환은 쓰지 않는다. 누가 뭐라고 하면 성능 최적화라고 해라

- DB 는 그때 그때 필요한 필드를 추가 해라. 정규화나 규칙은 생각조차 하지말아라. 누가 뭐라고 하면 이 또한 성능 최적화

- 패턴이나 아키텍처는 쓰지 않는다. 클래스로도 나누지 마라. N 중 포문과 if 문으로 절차적으로 작성해라,
누가 뭐라고 하면 이건 진짜 성능 최적화라고 해라

- 성능최적화는 그 어떠한 경우에도 금물, 누가 뭐라고 하면 요즘은 "컴파일러" 한테 맡기는게 대세라고 해라
```

## 휴먼 오토(Human Automation)

### 백엔드
백앤드는 모던 비즈니스의 핵심이다. 엄한데 힘빼지 말자
```
- 그건 DBA 가 해야 할 일이라고 해라.
- 그건 DevOps 엔지니어가 해야할 일이라고 해라.
- 그건 프론트엔드가 해야 할 일이라고 해라.
- 그건 기획팀에서 먼저 기획해야할 일이라고 해라.
- 그건 운영팀에게서 먼저 확인받아야할 일이라고 해라.
```

### 프론트엔드

프론트엔드는 유저가 마주하는 첫인상이다. 엄한데 힘빼지 말자
```
- 그건 백엔드가 해야 할 일이라고 해라.
- 그건 앱 개발자가 해야할 일이라고 해라.
- 그건 퍼블리셔가 해야 할 일이라고 해라.
- 그건 디자이너가 해야 할 일이라고 해라.
- 그건 유저가 해야 할 일이라고 해라.
```
### DBA

DBMS 개발자를 믿자

```
- N+1 문제가 자동으로 해결되지 않는 이 세상이 이상한거다. 개발자는 신경쓰지 말자.

- fetch join + paging 문제가 자동으로 해결되지 않는 이 세상이 이상한거다. 개발자는 신경쓰지 말자.

- slow query 문제가 자동으로 해결되지 않는 이 세상이 이상한거다. 개발자는 신경쓰지 말자.

- SQL / ORM 등에서 힘겹게 쿼링하는것보다 그냥 다 불러와서 map/reduce/filter 쓰는게 더 편하다.
```

### 머신러닝
요즘 기계는 대충 가르쳐도 알아서 잘 배운다.

```
- 데이터 하나가 소중한 시점에 Validation set 에 떼줄 데이터 따위 없다. 죄다 학습에 넣어버리자.

- 오버피팅이 일어나면 오히려 좋다. 누군가 딴지를 걸면 "실험 환경에서는 성능이 좋았는데요."라고 말하자.

- 성능이 너무 낮으면 데이터가 부족하기 때문이라고 하자.

- 모델이 너무 크다면 딥러닝이 원래 그런거라고 하자.

- 모델이 너무 작다면 모델 최적화의 결과라고 하자. 최적화의 부작용으로 성능이 조금 떨어질 수 있다는 점도 곁들여주면 좋다.

- training 이 너무 오래 걸린다고하면, 더 좋은 GPU 를 쓰면 된다고 하면서 NVIDIA DGX A100 같은 것을 보여주자.

- inference 가 너무 오래 걸린다고하면, 더 좋은 GPU 를 쓰면 된다고 하면서 NVIDIA V100 같은 것을 보여주자.

- Productization 은 DevOps, MLOps 엔지니어의 역할이다. 모델 리서처는 신경쓰지 말자.
```

## 기타
솔직히 좀 뇌절인듯
```
- 그 날 배운 기술은 그 날 회사 프로젝트에 적용해라. 누가 뭐라고 하면 그 날 배운 지식을 자랑하면된다

- 기술 스택을 공부 하지 말고, 해당 기술 스택의 단점리스트만 공부해라. 누가 해당 기술 스택을 물어보면, 단점을 들먹이자

- 도커같이 가상화가 조금이라도 들어간건 사용하지 말아라. 누가 뭐라고하면 네이티브 환경에서의 확인이 필요하다고 해라
```


# Reference
놀랍게도 꽤 많이 참고했다
- [유지보수하기 어렵게 코딩하는 방법: 평생 개발자로 먹고 살 수 있다](https://www.hanbit.co.kr/store/books/look.php?p_code=E2375873090)
- [애자일 선언](https://agilemanifesto.org/iso/ko/manifesto.html)
- [프로그래밍의 정석](http://www.yes24.com/Product/Goods/55254076)
- [7가지 동시성 모델](http://www.yes24.com/Product/Goods/29331038)
- [폴리글랏 프로그래밍](http://www.yes24.com/Product/Goods/12204890)
- [클린 코드](http://www.yes24.com/Product/Goods/11681152)
- [클린 아키텍쳐](http://www.yes24.com/Product/Goods/77283734)
- [클로저 프로그래밍의 즐거움](http://www.yes24.com/Product/Goods/24555451)
- [프로그래밍 스칼라](http://www.yes24.com/Product/Goods/27767797)
- [FSharp Fun and Profit 블로그](https://fsharpforfunandprofit.com/)
- [로버트 C 마틴 블로그](https://blog.cleancoder.com/)
- [마틴파울러 블로그](https://martinfowler.com/)
- 그외 언젠가 한번쯤 읽어본 책들 다수
- 그외 언젠가 한번쯤 읽어볼 책들 다수

# Contributing
더 좋은 주둥아리 방법을 알고있다면 PR 을 날려주세요

# License
JDD 는 어떠한 제약조건도 없습니다.
단 책임도 지지 않습니다.