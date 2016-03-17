#ifndef OBSERVER_H_
#define OBSERVER_H_

namespace ddrf
{
	class Subject;

	class Observer
	{
		public:
			virtual ~Observer() = default;
			virtual auto update(Subject* s) -> void = 0;

		protected:
			Observer() = default;
	};
}



#endif /* OBSERVER_H_ */
